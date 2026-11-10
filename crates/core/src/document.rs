//! The open document: scene + selection + undo/redo + file state, and all object-mode operations.
//! Undo is snapshot based; snapshots are cheap because meshes/textures are `Arc`-shared.
use crate::armature::{ArmatureData, SkinBinding};
use crate::kinds::{CameraData, LightData, LightType};
use crate::mesh::Aabb;
use crate::{Collection, EditDomain, EditSession, Material, MeshData, Object, ObjectKind, Primitive, Scene, TextureData, Vertex};
use glam::{Mat4, Quat, Vec3};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OriginMode {
    /// center of the bounding box
    Geometry,
    /// average of all vertices
    Median,
    Cursor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlignMode {
    Min,
    Center,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlignTo {
    Active,
    Selection,
    Cursor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SameKind {
    Type,
    Collection,
    Material,
    Linked,
}

pub struct Document {
    pub scene: Scene,
    /// Ids of selected objects.
    pub selection: BTreeSet<u64>,
    /// The active object (shown in Properties). Usually part of the selection.
    pub active: Option<u64>,
    /// Collection new objects are added to.
    pub active_collection: Option<u64>,
    pub path: Option<PathBuf>,
    pub dirty: bool,
    /// Bumped on every change (including selection) so the renderer knows when to re-sync.
    pub version: u64,
    /// UI bookkeeping: a continuous edit (slider drag, typing) already has its undo snapshot.
    pub edit_open: bool,
    /// Active mesh edit session (Tab / Edit Mode).
    pub edit: Option<EditSession>,
    /// When true, the active armature is in bone Edit Mode (structure).
    pub armature_edit: bool,
    /// Armature Pose Mode; separate from Object Mode and Armature Edit Mode.
    pub pose_mode: bool,
    /// Sculpt Mode on the active mesh.
    pub sculpt_mode: bool,
    /// UV Editing Mode.
    pub uv_mode: bool,
    /// Texture Paint Mode.
    pub paint_mode: bool,
    /// Selected vertex indices for UV editing.
    pub uv_selection: std::collections::BTreeSet<u32>,
    rng: u32,
    undo: Vec<UndoSnap>,
    redo: Vec<UndoSnap>,
}

/// Full snapshot for undo/redo: scene + optional Edit Mode session.
#[derive(Clone)]
struct UndoSnap {
    scene: Scene,
    edit: Option<EditSession>,
}


const MAX_UNDO: usize = 200;

impl Document {
    /// New document with a Blender-like default scene: Cube + Sun Light + Camera.
    pub fn new() -> Self {
        let mut d = Self::from_scene(Scene::default(), None);

        // Default cube at origin
        let mut cube = Object::new("Cube", Primitive::Cube { size: 1.0 }.build());
        cube.primitive = Some(Primitive::Cube { size: 1.0 });
        d.scene.objects.push(cube);

        // Sun light above and slightly offset
        let mut light = Object::new(
            "Sun Light",
            MeshData::default(),
        );
        light.translation = Vec3::new(0.0, 4.0, 0.0);
        light.rotation = Quat::from_rotation_arc(Vec3::NEG_Z, Vec3::new(-0.4, -0.8, -0.4).normalize());
        light.kind = ObjectKind::Light(LightData::new(LightType::Sun));
        d.scene.objects.push(light);

        // Perspective camera looking at origin
        let eye = Vec3::new(4.0, 3.0, 6.0);
        let mut cam = Object::new("Camera", MeshData::default());
        cam.translation = eye;
        cam.rotation = Quat::from_mat4(&Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y).inverse());
        cam.kind = ObjectKind::Camera(CameraData::default());
        d.scene.objects.push(cam);

        d.scene.fix_ids();
        d.dirty = false;
        d
    }

    pub fn from_scene(mut scene: Scene, path: Option<PathBuf>) -> Self {
        scene.fix_ids();
        Self {
            scene,
            selection: BTreeSet::new(),
            active: None,
            active_collection: None,
            path,
            dirty: false,
            version: 1,
            edit_open: false,
            edit: None,
            armature_edit: false,
            pose_mode: false,
            sculpt_mode: false,
            uv_mode: false,
            paint_mode: false,
            uv_selection: std::collections::BTreeSet::new(),
            rng: 0x9E37_79B9,
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    pub fn touch(&mut self) {
        self.dirty = true;
        self.version += 1;
    }

    fn sel_changed(&mut self) {
        if self.active_armature().is_none() {
            self.armature_edit = false;
            self.pose_mode = false;
        }
        self.version += 1;
    }

    // ------------------------------------------------------------------ undo
    /// Call BEFORE mutating the scene.
    fn take_snap(&self) -> UndoSnap {
        UndoSnap {
            scene: self.scene.clone(),
            edit: self.edit.clone(),
        }
    }

    fn restore_snap(&mut self, snap: UndoSnap) {
        self.scene = snap.scene;
        self.edit = snap.edit;
    }

    pub fn checkpoint(&mut self) {
        self.undo.push(self.take_snap());
        if self.undo.len() > MAX_UNDO {
            self.undo.remove(0);
        }
        self.redo.clear();
        self.touch();
    }

    /// Forget the most recent snapshot without restoring it (used when a modal operation is cancelled).
    pub fn drop_last_checkpoint(&mut self) {
        self.undo.pop();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }
    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    pub fn undo(&mut self) -> bool {
        match self.undo.pop() {
            Some(prev) => {
                let current = self.take_snap();
                self.restore_snap(prev);
                self.redo.push(current);
                self.after_history();
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self) -> bool {
        match self.redo.pop() {
            Some(next) => {
                let current = self.take_snap();
                self.restore_snap(next);
                self.undo.push(current);
                self.after_history();
                true
            }
            None => false,
        }
    }

    fn after_history(&mut self) {
        let ids: HashSet<u64> = self.scene.objects.iter().map(|o| o.id).collect();
        self.selection.retain(|id| ids.contains(id));
        if self.active.map_or(false, |a| !ids.contains(&a)) {
            self.active = None;
        }
        if self.active_armature().is_none() {
            self.armature_edit = false;
            self.pose_mode = false;
        }
        // Keep Edit Mode only if the edited object still exists
        if let Some(e) = self.edit.as_ref() {
            if !ids.contains(&e.object_id) {
                self.edit = None;
            }
        }
        self.touch();
    }

    pub fn unique_name(&self, base: &str) -> String {
        let exists = |n: &str| self.scene.objects.iter().any(|o| o.name == n);
        if !exists(base) {
            return base.to_string();
        }
        let root = base.rsplit_once('.').filter(|(_, n)| n.chars().all(|c| c.is_ascii_digit())).map_or(base, |(r, _)| r);
        (1..).map(|i| format!("{root}.{i:03}")).find(|n| !exists(n)).unwrap()
    }

    // ------------------------------------------------------------------ selection
    pub fn active_index(&self) -> Option<usize> {
        self.active.and_then(|id| self.scene.index_of(id))
    }

    pub fn selected_indices(&self) -> Vec<usize> {
        self.scene.objects.iter().enumerate().filter(|(_, o)| self.selection.contains(&o.id)).map(|(i, _)| i).collect()
    }

    pub fn is_selected(&self, i: usize) -> bool {
        self.scene.objects.get(i).map_or(false, |o| self.selection.contains(&o.id))
    }

    pub fn select_only(&mut self, i: usize) {
        self.selection.clear();
        if let Some(o) = self.scene.objects.get(i) {
            self.selection.insert(o.id);
            self.active = Some(o.id);
        }
        self.sel_changed();
    }

    pub fn select_add(&mut self, i: usize) {
        if let Some(o) = self.scene.objects.get(i) {
            self.selection.insert(o.id);
            self.active = Some(o.id);
        }
        self.sel_changed();
    }

    pub fn select_remove(&mut self, i: usize) {
        if let Some(o) = self.scene.objects.get(i) {
            let id = o.id;
            self.selection.remove(&id);
            if self.active == Some(id) {
                self.active = self.selection.iter().next().copied();
            }
        }
        self.sel_changed();
    }

    /// Shift-click behaviour: add, or make active, or remove if it is already the active one.
    pub fn select_toggle(&mut self, i: usize) {
        let Some(o) = self.scene.objects.get(i) else { return };
        let id = o.id;
        if self.selection.contains(&id) {
            if self.active == Some(id) {
                self.select_remove(i);
            } else {
                self.active = Some(id);
                self.sel_changed();
            }
        } else {
            self.select_add(i);
        }
    }

    pub fn select_none(&mut self) {
        self.selection.clear();
        self.sel_changed();
    }

    pub fn select_all(&mut self) {
        let ids: Vec<u64> = (0..self.scene.objects.len()).filter(|&i| self.scene.is_visible(i)).map(|i| self.scene.objects[i].id).collect();
        for id in &ids {
            self.selection.insert(*id);
        }
        if self.active.is_none() {
            self.active = ids.first().copied();
        }
        self.sel_changed();
    }

    pub fn select_invert(&mut self) {
        let ids: Vec<u64> = (0..self.scene.objects.len()).filter(|&i| self.scene.is_visible(i)).map(|i| self.scene.objects[i].id).collect();
        for id in ids {
            if !self.selection.remove(&id) {
                self.selection.insert(id);
            }
        }
        self.sel_changed();
    }

    pub fn select_random(&mut self, ratio: f32) {
        let ids: Vec<u64> = (0..self.scene.objects.len()).filter(|&i| self.scene.is_visible(i)).map(|i| self.scene.objects[i].id).collect();
        for id in ids {
            self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
            if ((self.rng >> 8) as f32 / (1u32 << 24) as f32) < ratio {
                self.selection.insert(id);
            }
        }
        self.sel_changed();
    }

    pub fn select_children(&mut self) {
        let mut frontier: Vec<u64> = self.selection.iter().copied().collect();
        while let Some(id) = frontier.pop() {
            for c in self.scene.children_of(id) {
                let cid = self.scene.objects[c].id;
                if self.selection.insert(cid) {
                    frontier.push(cid);
                }
            }
        }
        self.sel_changed();
    }

    pub fn select_parent(&mut self) {
        let parents: Vec<u64> = self.selected_indices().iter().filter_map(|&i| self.scene.objects[i].parent).collect();
        if parents.is_empty() {
            return;
        }
        self.selection.clear();
        for p in &parents {
            self.selection.insert(*p);
        }
        self.active = parents.first().copied();
        self.sel_changed();
    }

    pub fn select_same(&mut self, k: SameKind) {
        let Some(a) = self.active_index() else { return };
        let ids: Vec<u64> = {
            let ao = &self.scene.objects[a];
            self.scene
                .objects
                .iter()
                .enumerate()
                .filter(|(i, o)| {
                    self.scene.is_visible(*i)
                        && match k {
                            SameKind::Type => std::mem::discriminant(&o.kind) == std::mem::discriminant(&ao.kind),
                            SameKind::Collection => o.collection == ao.collection,
                            SameKind::Material => o.material == ao.material,
                            SameKind::Linked => (o.link_group.is_some() && o.link_group == ao.link_group) || Arc::ptr_eq(&o.mesh, &ao.mesh),
                        }
                })
                .map(|(_, o)| o.id)
                .collect()
        };
        for id in ids {
            self.selection.insert(id);
        }
        self.sel_changed();
    }

    // ------------------------------------------------------------------ bounds
    pub fn selected_bounds(&self) -> Option<Aabb> {
        let w = self.scene.world_matrices();
        self.selected_indices().into_iter().filter_map(|i| self.scene.object_world_bounds(i, &w[i])).reduce(Aabb::union)
    }

    // ------------------------------------------------------------------ adding
    fn add_object_select(&mut self, mut o: Object) {
        o.id = self.scene.alloc_id();
        o.collection = self.active_collection.filter(|c| self.scene.collections.iter().any(|x| x.id == *c));
        let id = o.id;
        self.scene.objects.push(o);
        self.selection.clear();
        self.selection.insert(id);
        self.active = Some(id);
        self.sel_changed();
    }

    pub fn add_primitive(&mut self, p: Primitive) {
        self.checkpoint();
        let mut o = Object::new(self.unique_name(p.name()), p.build());
        o.translation = self.scene.cursor;
        o.primitive = Some(p);
        self.add_object_select(o);
    }

    /// Add a camera, light or empty.
    pub fn add_kind(&mut self, kind: ObjectKind) {
        self.checkpoint();
        let mut o = Object::new(self.unique_name(kind.default_name()), MeshData::default());
        o.translation = self.scene.cursor;
        match &kind {
            ObjectKind::Camera(_) => {
                let eye = Vec3::new(4.0, 3.0, 6.0);
                o.translation = eye;
                o.rotation = Quat::from_mat4(&Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y).inverse());
            }
            ObjectKind::Light(l) => match l.kind {
                LightType::Point => o.translation = Vec3::new(2.0, 3.0, 2.0),
                LightType::Sun => {
                    o.translation = Vec3::new(0.0, 4.0, 0.0);
                    o.rotation = Quat::from_rotation_arc(Vec3::NEG_Z, Vec3::new(-0.4, -0.8, -0.4).normalize());
                }
                LightType::Spot | LightType::Area => {
                    o.translation = Vec3::new(2.0, 4.0, 2.0);
                    o.rotation = Quat::from_rotation_arc(Vec3::NEG_Z, (-o.translation).normalize());
                }
            },
            _ => {}
        }
        o.kind = kind;
        self.add_object_select(o);
    }

    /// Upright textured quad (faces +Z) with the image's aspect ratio. Handy for sprites / references.
    pub fn add_image_plane(&mut self, name: &str, tex: TextureData) {
        self.checkpoint();
        let aspect = tex.width.max(1) as f32 / tex.height.max(1) as f32;
        let (hw, hh) = (aspect * 0.5, 0.5);
        let n = [0.0, 0.0, 1.0];
        let mesh = MeshData {
            name: name.to_string(),
            vertices: vec![
                Vertex { position: [-hw, hh, 0.0], normal: n, uv: [0.0, 0.0] },
                Vertex { position: [hw, hh, 0.0], normal: n, uv: [1.0, 0.0] },
                Vertex { position: [hw, -hh, 0.0], normal: n, uv: [1.0, 1.0] },
                Vertex { position: [-hw, -hh, 0.0], normal: n, uv: [0.0, 1.0] },
            ],
            indices: vec![0, 3, 2, 0, 2, 1],
            ..Default::default()
        };
        let mut texs: Vec<TextureData> = (*self.scene.textures).clone();
        let ti = texs.len();
        texs.push(tex);
        self.scene.textures = Arc::new(texs);
        let mat = Material {
            name: name.to_string(),
            base_color: [1.0, 1.0, 1.0, 1.0],
            roughness: 1.0,
            alpha_mode: crate::AlphaMode::Blend,
            double_sided: true,
            base_color_tex: Some(ti),
            ..Material::default()
        };
        self.scene.materials.push(mat);
        let mut o = Object::new(self.unique_name(name), mesh);
        o.material = Some(self.scene.materials.len() - 1);
        o.translation = self.scene.cursor + Vec3::new(0.0, 0.5, 0.0);
        self.add_object_select(o);
    }

    /// Merge another scene into this one (import).
    pub fn import_scene(&mut self, other: Scene) {
        if other.objects.is_empty() {
            return;
        }
        self.checkpoint();
        let mat_off = self.scene.materials.len();
        let tex_off = self.scene.textures.len();
        let mut tex: Vec<_> = (*self.scene.textures).clone();
        tex.extend(other.textures.iter().cloned());
        self.scene.textures = Arc::new(tex);
        for mut m in other.materials {
            m.offset_textures(tex_off);
            self.scene.materials.push(m);
        }
        let mut new_ids = Vec::new();
        for mut o in other.objects {
            o.material = o.material.map(|i| i + mat_off);
            for g in Arc::make_mut(&mut o.mesh).groups.iter_mut() {
                g.material = g.material.map(|i| i + mat_off);
            }
            o.name = self.unique_name(&o.name);
            o.id = self.scene.alloc_id();
            o.parent = None;
            o.collection = self.active_collection;
            new_ids.push(o.id);
            self.scene.objects.push(o);
        }
        self.selection = new_ids.iter().copied().collect();
        self.active = new_ids.last().copied();
        self.sel_changed();
    }

    // ------------------------------------------------------------------ delete / duplicate
    fn remove_objects(&mut self, ids: &HashSet<u64>) {
        let worlds = self.scene.world_matrices();
        for i in 0..self.scene.objects.len() {
            let o = &self.scene.objects[i];
            if ids.contains(&o.id) {
                continue;
            }
            if let Some(p) = o.parent {
                if ids.contains(&p) {
                    let (s, r, t) = worlds[i].to_scale_rotation_translation();
                    let o = &mut self.scene.objects[i];
                    o.parent = None;
                    o.scale = s;
                    o.rotation = r;
                    o.translation = t;
                }
            }
        }
        self.scene.objects.retain(|o| !ids.contains(&o.id));
        self.selection.retain(|id| !ids.contains(id));
        if self.active.map_or(false, |a| ids.contains(&a)) {
            self.active = None;
        }
        self.sel_changed();
    }

    pub fn delete_selected(&mut self) {
        let ids: HashSet<u64> = self.selection.iter().copied().collect();
        if ids.is_empty() {
            return;
        }
        self.checkpoint();
        self.remove_objects(&ids);
    }

    /// Duplicate the selection (linked = share parametric data). The copies become the selection.
    pub fn duplicate(&mut self, linked: bool) -> bool {
        let sel = self.selected_indices();
        if sel.is_empty() {
            return false;
        }
        self.checkpoint();
        let active_old = self.active;
        let mut map: HashMap<u64, u64> = HashMap::new();
        let mut news: Vec<u64> = Vec::new();
        for &i in &sel {
            let mut o = self.scene.objects[i].clone();
            let old = o.id;
            if linked {
                let g = match self.scene.objects[i].link_group {
                    Some(g) => g,
                    None => {
                        let g = self.scene.alloc_id();
                        self.scene.objects[i].link_group = Some(g);
                        g
                    }
                };
                o.link_group = Some(g);
            } else {
                o.mesh = Arc::new((*o.mesh).clone());
                o.link_group = None;
            }
            o.id = self.scene.alloc_id();
            o.name = self.unique_name(&o.name);
            map.insert(old, o.id);
            news.push(o.id);
            self.scene.objects.push(o);
        }
        for id in &news {
            if let Some(idx) = self.scene.index_of(*id) {
                if let Some(p) = self.scene.objects[idx].parent {
                    if let Some(&np) = map.get(&p) {
                        self.scene.objects[idx].parent = Some(np);
                    }
                }
            }
        }
        self.selection = news.iter().copied().collect();
        self.active = active_old.and_then(|a| map.get(&a).copied()).or(news.last().copied());
        self.sel_changed();
        true
    }

    pub fn make_single_user(&mut self) {
        let sel = self.selected_indices();
        if sel.is_empty() {
            return;
        }
        self.checkpoint();
        for i in sel {
            let o = &mut self.scene.objects[i];
            o.mesh = Arc::new((*o.mesh).clone());
            o.link_group = None;
        }
    }

    // ------------------------------------------------------------------ visibility
    pub fn hide_selected(&mut self) {
        let sel = self.selected_indices();
        if sel.is_empty() {
            return;
        }
        self.checkpoint();
        for i in sel {
            self.scene.objects[i].visible = false;
        }
        self.selection.clear();
        self.sel_changed();
    }

    pub fn hide_unselected(&mut self) {
        self.checkpoint();
        for o in self.scene.objects.iter_mut() {
            if !self.selection.contains(&o.id) {
                o.visible = false;
            }
        }
    }

    pub fn show_all(&mut self) {
        self.checkpoint();
        for o in self.scene.objects.iter_mut() {
            o.visible = true;
        }
        for c in self.scene.collections.iter_mut() {
            c.visible = true;
        }
    }

    // ------------------------------------------------------------------ parenting
    /// Make `child` a child of `parent` (None = unparent), keeping its world transform.
    pub fn set_parent_of(&mut self, child: usize, parent: Option<u64>) -> Result<(), String> {
        if let Some(pid) = parent {
            let Some(pi) = self.scene.index_of(pid) else { return Err("Parent not found".into()) };
            if pi == child || self.scene.is_ancestor(child, pi) {
                return Err("That would create a parent loop".into());
            }
        }
        self.checkpoint();
        let worlds = self.scene.world_matrices();
        let w = worlds[child];
        self.scene.objects[child].parent = parent;
        self.scene.set_world(child, w, &worlds);
        Ok(())
    }

    /// Selected objects become children of the active object.
    pub fn parent_set(&mut self) -> Result<(), String> {
        let Some(ai) = self.active_index() else { return Err("No active object (select the parent last)".into()) };
        let children: Vec<usize> = self.selected_indices().into_iter().filter(|&i| i != ai).collect();
        if children.is_empty() {
            return Err("Select the children first, then the parent last (active)".into());
        }
        self.checkpoint();
        let worlds = self.scene.world_matrices();
        let pid = self.scene.objects[ai].id;
        let mut n = 0;
        for ci in children {
            if self.scene.is_ancestor(ci, ai) {
                continue; // would create a loop
            }
            let w = worlds[ci];
            self.scene.objects[ci].parent = Some(pid);
            self.scene.set_world(ci, w, &worlds);
            n += 1;
        }
        if n == 0 {
            return Err("Nothing parented (parent loop)".into());
        }
        Ok(())
    }

    pub fn parent_clear(&mut self, keep_transform: bool) {
        let sel: Vec<usize> = self.selected_indices().into_iter().filter(|&i| self.scene.objects[i].parent.is_some()).collect();
        if sel.is_empty() {
            return;
        }
        self.checkpoint();
        let worlds = self.scene.world_matrices();
        for i in sel {
            self.scene.objects[i].parent = None;
            if keep_transform {
                self.scene.set_world(i, worlds[i], &worlds);
            }
        }
    }

    // ------------------------------------------------------------------ collections
    pub fn new_collection(&mut self) -> u64 {
        self.checkpoint();
        let mut n = self.scene.collections.len() + 1;
        let name = loop {
            let cand = format!("Collection {n}");
            if !self.scene.collections.iter().any(|c| c.name == cand) {
                break cand;
            }
            n += 1;
        };
        let id = self.scene.alloc_id();
        self.scene.collections.push(Collection { id, name, visible: true });
        self.active_collection = Some(id);
        id
    }

    pub fn delete_collection(&mut self, id: u64) {
        self.checkpoint();
        for o in self.scene.objects.iter_mut() {
            if o.collection == Some(id) {
                o.collection = None;
            }
        }
        self.scene.collections.retain(|c| c.id != id);
        if self.active_collection == Some(id) {
            self.active_collection = None;
        }
    }

    pub fn move_selected_to_collection(&mut self, c: Option<u64>) {
        let sel = self.selected_indices();
        if sel.is_empty() {
            return;
        }
        self.checkpoint();
        for i in sel {
            self.scene.objects[i].collection = c;
        }
    }

    // ------------------------------------------------------------------ transforms
    pub fn clear_transform(&mut self, loc: bool, rot: bool, scale: bool) {
        let sel = self.selected_indices();
        if sel.is_empty() {
            return;
        }
        self.checkpoint();
        for i in sel {
            let o = &mut self.scene.objects[i];
            if loc {
                o.translation = Vec3::ZERO;
            }
            if rot {
                o.rotation = Quat::IDENTITY;
            }
            if scale {
                o.scale = Vec3::ONE;
            }
        }
    }

    /// Keep the world transform of `i`'s children after `i` changed (old worlds given).
    fn keep_children_world(&mut self, i: usize, old_worlds: &[Mat4]) {
        let nw = self.scene.world_matrices();
        let id = self.scene.objects[i].id;
        for c in self.scene.children_of(id) {
            self.scene.set_world(c, old_worlds[c], &nw);
        }
    }

    /// Bake location / rotation / scale into the mesh (the object keeps looking the same).
    pub fn apply_transform(&mut self, loc: bool, rot: bool, scale: bool) {
        let sel: Vec<usize> = self.selected_indices().into_iter().filter(|&i| self.scene.objects[i].kind.is_mesh()).collect();
        if sel.is_empty() {
            return;
        }
        self.checkpoint();
        for i in sel {
            let old_worlds = self.scene.world_matrices();
            let o = &self.scene.objects[i];
            let old_local = o.matrix();
            let (mut t, mut r, mut s) = (o.translation, o.rotation, o.scale);
            if loc {
                t = Vec3::ZERO;
            }
            if rot {
                r = Quat::IDENTITY;
            }
            if scale {
                s = Vec3::ONE;
            }
            let new_local = Mat4::from_scale_rotation_translation(s, r, t);
            let vm = new_local.inverse() * old_local;
            let mesh = Arc::new(o.mesh.transformed(vm));
            let o = &mut self.scene.objects[i];
            o.mesh = mesh;
            o.translation = t;
            o.rotation = r;
            o.scale = s;
            o.primitive = None;
            o.link_group = None;
            self.keep_children_world(i, &old_worlds);
        }
    }

    /// Move the object origin (the mesh stays where it is in the world).
    pub fn set_origin(&mut self, mode: OriginMode) {
        let sel: Vec<usize> = self.selected_indices().into_iter().filter(|&i| self.scene.objects[i].kind.is_mesh()).collect();
        if sel.is_empty() {
            return;
        }
        self.checkpoint();
        for i in sel {
            let old_worlds = self.scene.world_matrices();
            let o = &self.scene.objects[i];
            let off = match mode {
                OriginMode::Geometry => o.mesh.bounds().map_or(Vec3::ZERO, |b| b.center()),
                OriginMode::Median => o.mesh.centroid(),
                OriginMode::Cursor => {
                    let w = old_worlds[i];
                    if w.determinant().abs() < 1e-12 { Vec3::ZERO } else { w.inverse().transform_point3(self.scene.cursor) }
                }
            };
            let new_t = o.matrix().transform_point3(off);
            let mesh = Arc::new(o.mesh.transformed(Mat4::from_translation(-off)));
            let o = &mut self.scene.objects[i];
            o.mesh = mesh;
            o.translation = new_t;
            o.primitive = None;
            o.link_group = None;
            self.keep_children_world(i, &old_worlds);
        }
    }

    /// Move the mesh so its bounding-box center sits on the object origin (the object moves visually).
    pub fn geometry_to_origin(&mut self) {
        let sel: Vec<usize> = self.selected_indices().into_iter().filter(|&i| self.scene.objects[i].kind.is_mesh()).collect();
        if sel.is_empty() {
            return;
        }
        self.checkpoint();
        for i in sel {
            let o = &mut self.scene.objects[i];
            let c = o.mesh.bounds().map_or(Vec3::ZERO, |b| b.center());
            o.mesh = Arc::new(o.mesh.transformed(Mat4::from_translation(-c)));
            o.primitive = None;
            o.link_group = None;
        }
    }

    // ------------------------------------------------------------------ align / snap
    pub fn align(&mut self, axis: usize, mode: AlignMode, to: AlignTo) {
        let sel = self.selected_indices();
        if sel.is_empty() {
            return;
        }
        let worlds = self.scene.world_matrices();
        let target: Option<(f32, f32, f32)> = {
            let b = |i: usize| self.scene.object_world_bounds(i, &worlds[i]);
            match to {
                AlignTo::Active => self.active_index().and_then(b).map(|bb| (bb.min[axis], bb.center()[axis], bb.max[axis])),
                AlignTo::Selection => sel.iter().filter_map(|&i| b(i)).reduce(Aabb::union).map(|bb| (bb.min[axis], bb.center()[axis], bb.max[axis])),
                AlignTo::Cursor => {
                    let c = self.scene.cursor[axis];
                    Some((c, c, c))
                }
            }
        };
        let Some((tmin, tc, tmax)) = target else { return };
        self.checkpoint();
        for i in self.scene.roots(&sel) {
            let Some(bb) = self.scene.object_world_bounds(i, &worlds[i]) else { continue };
            let delta = match mode {
                AlignMode::Min => tmin - bb.min[axis],
                AlignMode::Center => tc - bb.center()[axis],
                AlignMode::Max => tmax - bb.max[axis],
            };
            let mut d = Vec3::ZERO;
            d[axis] = delta;
            self.scene.set_world(i, Mat4::from_translation(d) * worlds[i], &worlds);
        }
    }

    fn move_roots_to(&mut self, target: Vec3, skip: Option<usize>) {
        let sel: Vec<usize> = self.selected_indices().into_iter().filter(|&i| Some(i) != skip).collect();
        if sel.is_empty() {
            return;
        }
        self.checkpoint();
        let worlds = self.scene.world_matrices();
        for i in self.scene.roots(&sel) {
            let mut w = worlds[i];
            w.w_axis = target.extend(1.0);
            self.scene.set_world(i, w, &worlds);
        }
    }

    pub fn selection_to_cursor(&mut self) {
        let c = self.scene.cursor;
        self.move_roots_to(c, None);
    }

    pub fn selection_to_active(&mut self) {
        let Some(a) = self.active_index() else { return };
        let t = self.scene.world_matrices()[a].w_axis.truncate();
        self.move_roots_to(t, Some(a));
    }

    pub fn cursor_to_selection(&mut self) {
        let sel = self.selected_indices();
        if sel.is_empty() {
            return;
        }
        let w = self.scene.world_matrices();
        let sum: Vec3 = sel.iter().map(|&i| w[i].w_axis.truncate()).sum();
        self.scene.cursor = sum / sel.len() as f32;
        self.sel_changed();
    }

    pub fn cursor_to_active(&mut self) {
        if let Some(a) = self.active_index() {
            self.scene.cursor = self.scene.world_matrices()[a].w_axis.truncate();
            self.sel_changed();
        }
    }

    pub fn cursor_to_origin(&mut self) {
        self.scene.cursor = Vec3::ZERO;
        self.sel_changed();
    }

    // ------------------------------------------------------------------ join / separate
    /// Merge the selected meshes into the active mesh object.

    /// Shade Smooth — average normals so curved meshes look smooth.
    pub fn shade_smooth_selected(&mut self) {
        let sel: Vec<usize> = self.selected_indices();
        if sel.is_empty() {
            return;
        }
        self.checkpoint();
        for i in sel {
            if !self.scene.objects[i].kind.is_mesh() {
                continue;
            }
            let mut mesh = (*self.scene.objects[i].mesh).clone();
            mesh.shade_smooth();
            self.scene.objects[i].mesh = std::sync::Arc::new(mesh);
            self.scene.objects[i].primitive = None; // baked shading
        }
        self.touch();
    }

    /// Shade Flat — per-face normals (faceted look).
    pub fn shade_flat_selected(&mut self) {
        let sel: Vec<usize> = self.selected_indices();
        if sel.is_empty() {
            return;
        }
        self.checkpoint();
        for i in sel {
            if !self.scene.objects[i].kind.is_mesh() {
                continue;
            }
            let mut mesh = (*self.scene.objects[i].mesh).clone();
            mesh.shade_flat();
            self.scene.objects[i].mesh = std::sync::Arc::new(mesh);
            self.scene.objects[i].primitive = None;
        }
        self.touch();
    }

    pub fn join(&mut self) -> Result<(), String> {
        let Some(ai) = self.active_index() else { return Err("No active object".into()) };
        if !self.scene.objects[ai].kind.is_mesh() {
            return Err("The active object must be a mesh".into());
        }
        let others: Vec<usize> = self.selected_indices().into_iter().filter(|&i| i != ai && self.scene.objects[i].kind.is_mesh()).collect();
        if others.is_empty() {
            return Err("Select at least two mesh objects (active one last)".into());
        }
        self.checkpoint();
        let worlds = self.scene.world_matrices();
        let inv_a = if worlds[ai].determinant().abs() < 1e-12 { Mat4::IDENTITY } else { worlds[ai].inverse() };
        let mut dst = (*self.scene.objects[ai].mesh).clone();
        dst.ensure_groups(self.scene.objects[ai].material);
        for &i in &others {
            let o = &self.scene.objects[i];
            dst.append(&o.mesh, inv_a * worlds[i], o.material);
        }
        dst.merge_adjacent_groups();
        let ids: HashSet<u64> = others.iter().map(|&i| self.scene.objects[i].id).collect();
        {
            let a = &mut self.scene.objects[ai];
            a.mesh = Arc::new(dst);
            a.primitive = None;
            a.link_group = None;
        }
        self.remove_objects(&ids);
        Ok(())
    }

    fn add_split_copy(&mut self, src: usize, mesh: MeshData, material: Option<usize>) -> u64 {
        let mut n = self.scene.objects[src].clone();
        n.id = self.scene.alloc_id();
        n.name = self.unique_name(&mesh.name);
        n.mesh = Arc::new(mesh);
        n.material = material;
        n.primitive = None;
        n.link_group = None;
        n.parent = self.scene.objects[src].parent;
        let id = n.id;
        self.scene.objects.push(n);
        id
    }

    /// Split each selected mesh into its disconnected pieces.
    pub fn separate_loose(&mut self) -> usize {
        let sel: Vec<usize> = self.selected_indices().into_iter().filter(|&i| self.scene.objects[i].kind.is_mesh()).collect();
        let jobs: Vec<(usize, Vec<MeshData>)> = sel.iter().map(|&i| (i, self.scene.objects[i].mesh.loose_parts())).filter(|(_, p)| p.len() > 1).collect();
        if jobs.is_empty() {
            return 0;
        }
        self.checkpoint();
        let mut count = 0;
        for (i, parts) in jobs {
            let mat = self.scene.objects[i].material;
            let mut it = parts.into_iter();
            let first = it.next().unwrap();
            self.scene.objects[i].mesh = Arc::new(first);
            self.scene.objects[i].primitive = None;
            self.scene.objects[i].link_group = None;
            for p in it {
                let id = self.add_split_copy(i, p, mat);
                self.selection.insert(id);
                count += 1;
            }
        }
        self.sel_changed();
        count
    }

    /// Split each selected multi-material mesh into one object per material.
    pub fn separate_material(&mut self) -> usize {
        let sel: Vec<usize> = self.selected_indices().into_iter().filter(|&i| self.scene.objects[i].kind.is_mesh() && !self.scene.objects[i].mesh.groups.is_empty()).collect();
        let jobs: Vec<(usize, Vec<(Option<usize>, MeshData)>)> =
            sel.iter().map(|&i| (i, self.scene.objects[i].mesh.split_by_material())).filter(|(_, p)| p.len() > 1).collect();
        if jobs.is_empty() {
            // single material: just flatten groups
            return 0;
        }
        self.checkpoint();
        let mut count = 0;
        for (i, parts) in jobs {
            let mut it = parts.into_iter();
            let (m0, first) = it.next().unwrap();
            {
                let o = &mut self.scene.objects[i];
                o.mesh = Arc::new(first);
                o.material = m0;
                o.primitive = None;
                o.link_group = None;
            }
            for (m, p) in it {
                let id = self.add_split_copy(i, p, m);
                self.selection.insert(id);
                count += 1;
            }
        }
        self.sel_changed();
        count
    }

    // ------------------------------------------------------------------ Edit Mode
    pub fn in_edit_mode(&self) -> bool {
        self.edit.is_some() || self.armature_edit
    }



    pub fn enter_paint_mode(&mut self) -> Result<(), String> {
        if self.paint_mode {
            return Ok(());
        }
        if self.edit.is_some() {
            self.exit_edit_mode(true);
        }
        if self.sculpt_mode {
            self.exit_sculpt_mode();
        }
        if self.uv_mode {
            self.exit_uv_mode();
        }
        self.armature_edit = false;
        self.pose_mode = false;
        let id = self.active.or_else(|| self.selection.iter().copied().next())
            .ok_or_else(|| "Select a mesh object first".to_string())?;
        let obj = self.scene.objects.iter().find(|o| o.id == id)
            .ok_or_else(|| "Object not found".to_string())?;
        if !matches!(obj.kind, crate::ObjectKind::Mesh) {
            return Err("Only mesh objects support Texture Paint".into());
        }
        let has_usable_uvs = !obj.mesh.vertices.is_empty() && obj.mesh.vertices.iter().all(|v| {
            v.uv.iter().all(|c| c.is_finite())
        }) && obj.mesh.indices.len() >= 3 && {
            let mut min = glam::Vec2::splat(f32::INFINITY);
            let mut max = glam::Vec2::splat(f32::NEG_INFINITY);
            for vertex in &obj.mesh.vertices {
                let uv = glam::Vec2::from(vertex.uv);
                if uv.is_finite() { min = min.min(uv); max = max.max(uv); }
            }
            (max - min).length_squared() > 1.0e-10
        };
        if !has_usable_uvs {
            return Err("This mesh has no usable UV map. Unwrap it in UV Editing before painting.".into());
        }
        self.checkpoint();
        self.selection.clear();
        self.selection.insert(id);
        self.active = Some(id);
        self.ensure_paint_texture()?;
        self.paint_mode = true;
        Ok(())
    }

    pub fn exit_paint_mode(&mut self) {
        self.paint_mode = false;
    }

    pub fn toggle_paint_mode(&mut self) -> Result<(), String> {
        if self.paint_mode {
            self.exit_paint_mode();
            Ok(())
        } else {
            self.enter_paint_mode()
        }
    }

    /// Ensure active object material has a base color texture to paint into.
    pub fn ensure_paint_texture(&mut self) -> Result<(), String> {
        let id = self.active.ok_or_else(|| "No active object".to_string())?;
        let obj = self.scene.objects.iter().find(|o| o.id == id)
            .ok_or_else(|| "Object not found".to_string())?;
        let mat_idx = obj.material.unwrap_or(0);
        // Need mutable scene textures + materials
        let has_tex = self.scene.materials.get(mat_idx)
            .and_then(|m| m.base_color_tex)
            .and_then(|texture_index| self.scene.textures.get(texture_index))
            .is_some();
        if has_tex {
            return Ok(());
        }
        // Create a new paintable texture
        let w = 512u32;
        let h = 512u32;
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        // Light gray fill with subtle checker so paint is visible
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let c = if ((x / 32) + (y / 32)) % 2 == 0 { 200 } else { 180 };
                rgba[i] = c;
                rgba[i + 1] = c;
                rgba[i + 2] = c;
                rgba[i + 3] = 255;
            }
        }
        let tex = crate::TextureData {
            name: "Paint Atlas".into(),
            width: w,
            height: h,
            rgba8: rgba,
        };
        let mut textures = (*self.scene.textures).clone();
        let tex_idx = textures.len();
        textures.push(tex);
        self.scene.textures = std::sync::Arc::new(textures);
        // Assign to material
        if self.scene.materials.is_empty() {
            self.scene.materials.push(crate::Material::default());
        }
        let mi = mat_idx.min(self.scene.materials.len() - 1);
        self.scene.materials[mi].base_color_tex = Some(tex_idx);
        // Point object at that material
        if let Some(obj) = self.scene.objects.iter_mut().find(|o| o.id == id) {
            obj.material = Some(mi);
        }
        self.version = self.version.wrapping_add(1);
        self.dirty = true;
        Ok(())
    }

    /// Paint one soft brush stamp at UV (0..1).
    pub fn paint_at_uv(&mut self, uv: glam::Vec2, color: [f32; 4], radius_uv: f32, strength: f32) -> bool {
        self.paint_stroke_uv(&[uv], color, radius_uv, strength)
    }

    /// Paint a sequence of UV samples as one image update. Keeping the texture clone
    /// outside the sample loop avoids a full image copy and GPU re-upload per dab.
    pub fn paint_stroke_uv(&mut self, uvs: &[glam::Vec2], color: [f32; 4], radius_uv: f32, strength: f32) -> bool {
        if uvs.is_empty() { return false; }
        let id = match self.active { Some(i) => i, None => return false };
        let mat_idx = self.scene.objects.iter().find(|o| o.id == id)
            .and_then(|o| o.material).unwrap_or(0);
        let tex_idx = match self.scene.materials.get(mat_idx).and_then(|m| m.base_color_tex) {
            Some(t) => t,
            None => return false,
        };
        let mut textures = (*self.scene.textures).clone();
        let Some(tex) = textures.get_mut(tex_idx) else { return false };
        let (w, h) = (tex.width as i32, tex.height as i32);
        if w < 1 || h < 1 || tex.rgba8.len() != (w as usize * h as usize * 4) { return false; }

        let radius_uv = radius_uv.clamp(0.0001, 1.0);
        let radius_px = (radius_uv * w.min(h) as f32).max(1.0);
        let radius_i = radius_px.ceil() as i32;
        let strength = strength.clamp(0.0, 1.0);
        let src = [
            (color[0].clamp(0.0, 1.0) * 255.0).round() as u8,
            (color[1].clamp(0.0, 1.0) * 255.0).round() as u8,
            (color[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            (color[3].clamp(0.0, 1.0) * 255.0).round() as u8,
        ];
        let mut changed = false;
        for uv in uvs {
            // UVs outside the active image are not painted (no accidental edge smearing).
            if !uv.is_finite() || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 { continue; }
            let cx = uv.x * (w - 1) as f32;
            let cy = uv.y * (h - 1) as f32;
            let x0 = (cx.floor() as i32 - radius_i).max(0);
            let y0 = (cy.floor() as i32 - radius_i).max(0);
            let x1 = (cx.ceil() as i32 + radius_i).min(w - 1);
            let y1 = (cy.ceil() as i32 + radius_i).min(h - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let d = (dx * dx + dy * dy).sqrt();
                    if d > radius_px { continue; }
                    let t = (1.0 - d / radius_px).clamp(0.0, 1.0);
                    let falloff = t * t * (3.0 - 2.0 * t);
                    let alpha = (falloff * strength * (src[3] as f32 / 255.0)).clamp(0.0, 1.0);
                    if alpha <= 0.0 { continue; }
                    let i = ((y * w + x) * 4) as usize;
                    for channel in 0..3 {
                        let dst = tex.rgba8[i + channel] as f32;
                        tex.rgba8[i + channel] = (dst * (1.0 - alpha) + src[channel] as f32 * alpha)
                            .round().clamp(0.0, 255.0) as u8;
                    }
                    tex.rgba8[i + 3] = (tex.rgba8[i + 3] as f32 * (1.0 - alpha) + 255.0 * alpha)
                        .round().clamp(0.0, 255.0) as u8;
                    changed = true;
                }
            }
        }
        if changed {
            self.scene.textures = std::sync::Arc::new(textures);
            self.version = self.version.wrapping_add(1);
            self.dirty = true;
        }
        changed
    }

    pub fn enter_uv_mode(&mut self) -> Result<(), String> {
        if self.uv_mode {
            return Ok(());
        }
        if self.sculpt_mode {
            self.exit_sculpt_mode();
        }
        self.armature_edit = false;
        self.pose_mode = false;
        let id = self.active.or_else(|| self.selection.iter().copied().next())
            .ok_or_else(|| "Select a mesh object first".to_string())?;
        let obj = self.scene.objects.iter().find(|o| o.id == id)
            .ok_or_else(|| "Object not found".to_string())?;
        if !matches!(obj.kind, crate::ObjectKind::Mesh) {
            return Err("Only mesh objects can enter UV Editing".into());
        }
        self.checkpoint();
        self.selection.clear();
        self.selection.insert(id);
        self.active = Some(id);
        // UV workspace uses Edit Mode on the 3D side (like Blender)
        if self.edit.is_none() {
            if let Err(e) = self.enter_edit_mode() {
                return Err(e);
            }
        }
        self.uv_selection.clear();
        self.uv_mode = true;
        Ok(())
    }

    pub fn exit_uv_mode(&mut self) {
        self.uv_mode = false;
        self.uv_selection.clear();
        // Leave Edit Mode when leaving UV workspace
        if self.edit.is_some() {
            self.exit_edit_mode(true);
        }
    }

    pub fn toggle_uv_mode(&mut self) -> Result<(), String> {
        if self.uv_mode {
            self.exit_uv_mode();
            Ok(())
        } else {
            self.enter_uv_mode()
        }
    }

    fn active_mesh_mut(&mut self) -> Option<&mut crate::MeshData> {
        // Prefer live Edit Mode mesh so UV ops stay in sync with the 3D edit view
        if self.edit.is_some() {
            return self.edit.as_mut().map(|e| &mut e.mesh);
        }
        let id = self.active?;
        let obj = self.scene.objects.iter_mut().find(|o| o.id == id)?;
        let mesh = std::sync::Arc::make_mut(&mut obj.mesh);
        obj.primitive = None;
        Some(mesh)
    }

    fn after_uv_mesh_edit(&mut self) {
        // Push edit-session mesh back to the scene object and rebuild selection indices
        if let Some(session) = self.edit.as_mut() {
            let id = session.object_id;
            // Topology may have changed (split faces) — clear stale selection
            session.selection.clear();
            let mesh = session.mesh.clone();
            if let Some(obj) = self.scene.objects.iter_mut().find(|o| o.id == id) {
                obj.mesh = std::sync::Arc::new(mesh);
                obj.primitive = None;
            }
        } else if let Some(id) = self.active {
            // Already wrote through active_mesh_mut to object
            let _ = id;
        }
        self.uv_selection.clear();
        self.version = self.version.wrapping_add(1);
        self.dirty = true;
    }

    pub fn uv_unwrap_smart(&mut self) {
        self.checkpoint();
        if let Some(m) = self.active_mesh_mut() {
            crate::unwrap_smart(m);
        }
        self.after_uv_mesh_edit();
    }
    pub fn uv_unwrap_cube(&mut self) {
        self.checkpoint();
        if let Some(m) = self.active_mesh_mut() {
            crate::unwrap_cube(m);
        }
        self.after_uv_mesh_edit();
    }
    pub fn uv_unwrap_planar(&mut self) {
        self.checkpoint();
        if let Some(m) = self.active_mesh_mut() {
            crate::unwrap_planar(m);
        }
        self.after_uv_mesh_edit();
    }
    pub fn uv_unwrap_reset(&mut self) {
        self.checkpoint();
        if let Some(m) = self.active_mesh_mut() {
            crate::unwrap_reset(m);
        }
        self.after_uv_mesh_edit();
    }
    pub fn uv_pack(&mut self) {
        self.checkpoint();
        if let Some(m) = self.active_mesh_mut() {
            crate::pack_islands(m, 0.02);
        }
        self.after_uv_mesh_edit();
    }
    /// Apply a UV transform during an active pointer drag without creating one
    /// undo entry per frame. The live edit mesh and scene copy stay synchronized.
    fn sync_uv_drag_mesh(&mut self) {
        if let Some(session) = self.edit.as_ref() {
            let id = session.object_id;
            let mesh = session.mesh.clone();
            if let Some(obj) = self.scene.objects.iter_mut().find(|o| o.id == id) {
                obj.mesh = std::sync::Arc::new(mesh);
                obj.primitive = None;
            }
        }
        self.version = self.version.wrapping_add(1);
        self.dirty = true;
    }

    pub fn uv_translate_live(&mut self, dx: f32, dy: f32) {
        let selected = self.uv_selection.clone();
        if selected.is_empty() { return; }
        if let Some(mesh) = self.active_mesh_mut() {
            crate::uv_translate(mesh, &selected, glam::Vec2::new(dx, dy));
        }
        self.sync_uv_drag_mesh();
    }

    pub fn uv_scale_live(&mut self, factor: f32) {
        let selected = self.uv_selection.clone();
        if selected.is_empty() { return; }
        let pivot = if let Some(session) = self.edit.as_ref() {
            crate::uv_selection_center(&session.mesh, &selected)
        } else if let Some(id) = self.active {
            self.scene.objects.iter().find(|o| o.id == id)
                .map(|o| crate::uv_selection_center(&o.mesh, &selected))
                .unwrap_or(glam::Vec2::splat(0.5))
        } else { glam::Vec2::splat(0.5) };
        if let Some(mesh) = self.active_mesh_mut() {
            crate::uv_scale(mesh, &selected, pivot, factor);
        }
        self.sync_uv_drag_mesh();
    }

    pub fn uv_rotate_live(&mut self, angle: f32) {
        let selected = self.uv_selection.clone();
        if selected.is_empty() { return; }
        let pivot = if let Some(session) = self.edit.as_ref() {
            crate::uv_selection_center(&session.mesh, &selected)
        } else if let Some(id) = self.active {
            self.scene.objects.iter().find(|o| o.id == id)
                .map(|o| crate::uv_selection_center(&o.mesh, &selected))
                .unwrap_or(glam::Vec2::splat(0.5))
        } else { glam::Vec2::splat(0.5) };
        if let Some(mesh) = self.active_mesh_mut() {
            crate::uv_rotate(mesh, &selected, pivot, angle);
        }
        self.sync_uv_drag_mesh();
    }

    pub fn uv_translate(&mut self, dx: f32, dy: f32) {
        if self.uv_selection.is_empty() {
            return;
        }
        let sel = self.uv_selection.clone();
        if let Some(m) = self.active_mesh_mut() {
            crate::uv_translate(m, &sel, glam::Vec2::new(dx, dy));
        }
        self.after_uv_mesh_edit();
    }
    pub fn uv_scale_sel(&mut self, factor: f32) {
        if self.uv_selection.is_empty() {
            return;
        }
        let sel = self.uv_selection.clone();
        let pivot = {
            let id = self.active;
            let obj = self.scene.objects.iter().find(|o| Some(o.id) == id);
            obj.map(|o| crate::uv_selection_center(&o.mesh, &sel))
                .unwrap_or(glam::Vec2::splat(0.5))
        };
        if let Some(m) = self.active_mesh_mut() {
            crate::uv_scale(m, &sel, pivot, factor);
        }
        self.after_uv_mesh_edit();
    }
    pub fn uv_rotate_sel(&mut self, angle: f32) {
        if self.uv_selection.is_empty() {
            return;
        }
        let sel = self.uv_selection.clone();
        let pivot = {
            let id = self.active;
            let obj = self.scene.objects.iter().find(|o| Some(o.id) == id);
            obj.map(|o| crate::uv_selection_center(&o.mesh, &sel))
                .unwrap_or(glam::Vec2::splat(0.5))
        };
        if let Some(m) = self.active_mesh_mut() {
            crate::uv_rotate(m, &sel, pivot, angle);
        }
        self.after_uv_mesh_edit();
    }
    pub fn uv_select_all(&mut self) {
        let id = self.active;
        if let Some(obj) = self.scene.objects.iter().find(|o| Some(o.id) == id) {
            self.uv_selection = (0..obj.mesh.vertices.len() as u32).collect();
        }
    }
    pub fn uv_select_none(&mut self) {
        self.uv_selection.clear();
    }

    pub fn enter_sculpt_mode(&mut self) -> Result<(), String> {
        if self.sculpt_mode {
            return Ok(());
        }
        if self.edit.is_some() {
            self.exit_edit_mode(true);
        }
        self.armature_edit = false;
        self.pose_mode = false;
        let id = self.active.or_else(|| self.selection.iter().copied().next())
            .ok_or_else(|| "Select a mesh object first".to_string())?;
        let obj = self.scene.objects.iter().find(|o| o.id == id)
            .ok_or_else(|| "Object not found".to_string())?;
        if !matches!(obj.kind, crate::ObjectKind::Mesh) {
            return Err("Only mesh objects can enter Sculpt Mode".into());
        }
        if obj.mesh.vertices.is_empty() {
            return Err("Mesh has no vertices".into());
        }
        self.checkpoint();
        self.selection.clear();
        self.selection.insert(id);
        self.active = Some(id);
        self.sculpt_mode = true;
        Ok(())
    }

    pub fn exit_sculpt_mode(&mut self) {
        self.sculpt_mode = false;
    }

    pub fn toggle_sculpt_mode(&mut self) -> Result<(), String> {
        if self.sculpt_mode {
            self.exit_sculpt_mode();
            Ok(())
        } else {
            self.enter_sculpt_mode()
        }
    }

    /// Apply a sculpt brush sample in world-space ray; mutates active mesh.
    pub fn sculpt_stroke(
        &mut self,
        origin: glam::Vec3,
        dir: glam::Vec3,
        settings: &crate::SculptSettings,
        grab_delta_world: Option<glam::Vec3>,
    ) -> bool {
        let Some(id) = self.active else { return false };
        let Some(idx) = self.scene.objects.iter().position(|o| o.id == id) else { return false };
        let world = self.scene.world_matrices()[idx];
        let inv = world.inverse();
        let local_o = inv.transform_point3(origin);
        let local_d = inv.transform_vector3(dir).normalize_or_zero();
        let mesh_ref = self.scene.objects[idx].mesh.clone();
        let hit = crate::raycast_mesh(&mesh_ref, local_o, local_d);
        let Some((hit_p, hit_n, _)) = hit else { return false };
        let local_grab = grab_delta_world.map(|d| inv.transform_vector3(d));
        // Clone mesh, mutate, write back as new Arc
        let mut mesh = (*mesh_ref).clone();
        if settings.dyntopo {
            let _ = crate::dyntopo_refine(&mut mesh, hit_p, settings.radius, settings.dyntopo_detail);
        }
        crate::apply_brush(&mut mesh, settings, hit_p, hit_n, local_grab);
        if let Some(obj) = self.scene.objects.iter_mut().find(|o| o.id == id) {
            obj.mesh = std::sync::Arc::new(mesh);
            obj.primitive = None;
        }
        self.version = self.version.wrapping_add(1);
        self.dirty = true;
        true
    }

    pub fn enter_edit_mode(&mut self) -> Result<(), String> {
        if self.edit.is_some() {
            return Ok(());
        }
        let id = self.active.or_else(|| self.selection.iter().copied().next())
            .ok_or_else(|| "Select a mesh object first".to_string())?;
        let idx = self.scene.objects.iter().position(|o| o.id == id)
            .ok_or_else(|| "Object not found".to_string())?;
        let obj = &self.scene.objects[idx];
        if matches!(obj.kind, ObjectKind::Armature(_)) {
            self.pose_mode = false;
            self.armature_edit = true;
            self.checkpoint();
            return Ok(());
        }
        if !matches!(obj.kind, ObjectKind::Mesh) {
            return Err("Only mesh objects (or Armatures) can enter Edit Mode".into());
        }
        if obj.mesh.vertices.is_empty() {
            return Err("Mesh has no vertices".into());
        }
        let mesh = (*obj.mesh).clone();
        // Focus selection on this object
        self.checkpoint(); // so Ctrl+Z can leave Edit Mode / restore pre-edit scene
        self.selection.clear();
        self.selection.insert(id);
        self.active = Some(id);
        let mut session = EditSession::new(id, mesh);
        session.weld_coincident(1e-5);
        // Push welded mesh to the scene immediately
        if let Some(obj) = self.scene.objects.iter_mut().find(|o| o.id == id) {
            obj.mesh = std::sync::Arc::new(session.mesh.clone());
            obj.primitive = None;
        }
        self.edit = Some(session);
        Ok(())
    }

    /// Leave Edit Mode. `apply` keeps edits; otherwise restores the original mesh.
    pub fn exit_edit_mode(&mut self, apply: bool) {
        let Some(session) = self.edit.take() else { return };
        if apply {
            // Mesh was already live-synced during edits; keep current geometry.
            if let Some(obj) = self.scene.objects.iter_mut().find(|o| o.id == session.object_id) {
                obj.mesh = std::sync::Arc::new(session.mesh);
            }
        } else if let Some(obj) = self.scene.objects.iter_mut().find(|o| o.id == session.object_id) {
            // Cancel: restore mesh from before entering Edit Mode
            obj.mesh = std::sync::Arc::new(session.original);
        }
        self.version = self.version.wrapping_add(1);
        self.dirty = true;
    }

    pub fn toggle_edit_mode(&mut self) -> Result<(), String> {
        // Tab leaves Pose Mode for Object Mode; the mode selector can switch
        // directly to Armature Edit Mode when that is explicitly requested.
        if self.pose_mode {
            self.pose_mode = false;
            return Ok(());
        }
        if self.armature_edit {
            self.armature_edit = false;
            self.pose_mode = false;
            return Ok(());
        }
        if self.edit.is_some() {
            self.exit_edit_mode(true);
            Ok(())
        } else {
            self.enter_edit_mode()
        }
    }

    pub fn set_edit_domain(&mut self, domain: EditDomain) {
        if let Some(e) = self.edit.as_mut() {
            e.selection.domain = domain;
            e.selection.clear();
        }
    }

    /// Push working mesh to the scene object (live preview while editing).
    pub fn sync_edit_mesh(&mut self) {
        let Some(session) = self.edit.as_ref() else { return };
        let id = session.object_id;
        let mesh = session.mesh.clone();
        if let Some(obj) = self.scene.objects.iter_mut().find(|o| o.id == id) {
            obj.mesh = std::sync::Arc::new(mesh);
            obj.primitive = None; // stop parametric rebuild overwriting edits
        }
        self.version = self.version.wrapping_add(1);
        self.dirty = true;
    }

    pub fn edit_select_all(&mut self) {
        let Some(e) = self.edit.as_mut() else { return };
        match e.selection.domain {
            EditDomain::Vertex => {
                let n = e.mesh.vertices.len() as u32;
                e.selection.select_all_verts(n);
            }
            EditDomain::Edge => {
                let edges = e.build_edges();
                e.selection.select_all_edges(&edges);
            }
            EditDomain::Face => {
                let n = e.face_count();
                e.selection.select_all_faces(n);
            }
        }
    }

    pub fn edit_select_none(&mut self) {
        if let Some(e) = self.edit.as_mut() {
            e.selection.clear();
        }
    }

    pub fn edit_apply_pick(&mut self, pick: crate::edit::EditPick, extend: bool) {
        let Some(e) = self.edit.as_mut() else { return };
        if !extend {
            e.selection.clear();
        }
        match pick {
            crate::edit::EditPick::Vertex(i) => {
                if extend && e.selection.verts.contains(&i) {
                    e.selection.verts.remove(&i);
                } else {
                    e.selection.verts.insert(i);
                    e.expand_vert_selection_by_position(1e-5);
                }
            }
            crate::edit::EditPick::Edge(key) => {
                if extend && e.selection.edges.contains(&key) {
                    e.selection.edges.remove(&key);
                } else {
                    e.selection.edges.insert(key);
                }
            }
            crate::edit::EditPick::Face(f) => {
                if extend && e.selection.faces.contains(&f) {
                    e.selection.faces.remove(&f);
                } else {
                    e.selection.faces.insert(f);
                }
            }
        }
    }
    pub fn edit_delete(&mut self) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.delete_selection();
        }
        self.sync_edit_mesh();
    }

    pub fn edit_extrude(&mut self, amount: f32) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.extrude_faces(amount);
        }
        self.sync_edit_mesh();
    }

    pub fn edit_merge_verts(&mut self) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.merge_verts_at_center();
        }
        self.sync_edit_mesh();
    }

    pub fn edit_inset(&mut self, factor: f32) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.inset_faces(factor);
        }
        self.sync_edit_mesh();
    }

    pub fn edit_bevel(&mut self, amount: f32) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.bevel(amount);
        }
        self.sync_edit_mesh();
    }

    pub fn edit_smooth(&mut self, factor: f32) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.smooth_verts(factor);
        }
        self.sync_edit_mesh();
    }

    pub fn edit_shrink_fatten(&mut self, amount: f32) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.shrink_fatten(amount);
        }
        self.sync_edit_mesh();
    }

    pub fn edit_shear(&mut self, axis: usize, amount: f32) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.shear(axis, amount);
        }
        self.sync_edit_mesh();
    }

    pub fn edit_edge_slide(&mut self, amount: f32) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.edge_slide(amount);
        }
        self.sync_edit_mesh();
    }

    pub fn edit_rip(&mut self) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.rip_region();
        }
        self.sync_edit_mesh();
    }

    pub fn edit_loop_cut(&mut self) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.loop_cut();
        }
        self.sync_edit_mesh();
    }

    pub fn edit_spin(&mut self, steps: u32, angle_deg: f32) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.spin(steps, angle_deg);
        }
        self.sync_edit_mesh();
    }

    pub fn edit_knife_bisect(&mut self, plane_point: glam::Vec3, plane_normal: glam::Vec3) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.knife_bisect(plane_point, plane_normal);
        }
        self.sync_edit_mesh();
    }

    pub fn edit_poly_build(&mut self, amount: f32) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.poly_build(amount);
        }
        self.sync_edit_mesh();
    }

    pub fn edit_duplicate(&mut self) {
        if self.edit.is_none() { return; }
        self.checkpoint();
        if let Some(e) = self.edit.as_mut() {
            e.duplicate_faces(glam::Vec3::new(0.1, 0.1, 0.0));
        }
        self.sync_edit_mesh();
    }

    pub fn edit_invert_selection(&mut self) {
        let Some(e) = self.edit.as_mut() else { return };
        match e.selection.domain {
            EditDomain::Vertex => {
                let n = e.mesh.vertices.len() as u32;
                e.selection.invert_verts(n);
            }
            EditDomain::Edge => {
                let all = e.build_edges();
                let cur = e.selection.edges.clone();
                e.selection.edges = all.into_iter().filter(|x| !cur.contains(x)).collect();
            }
            EditDomain::Face => {
                let n = e.face_count();
                let cur = e.selection.faces.clone();
                e.selection.faces = (0..n).filter(|f| !cur.contains(f)).collect();
            }
        }
    }

    // ------------------------------------------------------------------ Armature
    pub fn active_armature_mut(&mut self) -> Option<&mut ArmatureData> {
        let id = self.active?;
        let obj = self.scene.objects.iter_mut().find(|o| o.id == id)?;
        match &mut obj.kind {
            ObjectKind::Armature(a) => Some(a),
            _ => None,
        }
    }

    pub fn active_armature(&self) -> Option<&ArmatureData> {
        let id = self.active?;
        let obj = self.scene.objects.iter().find(|o| o.id == id)?;
        match &obj.kind {
            ObjectKind::Armature(a) => Some(&a),
            _ => None,
        }
    }

    pub fn enter_armature_edit(&mut self) -> Result<(), String> {
        if self.active_armature().is_none() {
            return Err("Select an Armature object".into());
        }
        self.checkpoint();
        self.pose_mode = false;
        self.armature_edit = true;
        Ok(())
    }

    pub fn exit_armature_edit(&mut self) {
        self.armature_edit = false;
        self.pose_mode = false;
    }

    pub fn toggle_pose_mode(&mut self) -> Result<(), String> {
        if self.active_armature().is_none() {
            return Err("Select an Armature object".into());
        }
        if self.armature_edit {
            self.armature_edit = false;
            self.pose_mode = true;
        } else {
            self.pose_mode = !self.pose_mode;
        }
        self.touch();
        Ok(())
    }

    pub fn armature_extrude(&mut self) {
        self.checkpoint();
        if let Some(a) = self.active_armature_mut() {
            a.extrude_selected();
        }
        self.touch();
    }

    pub fn armature_add_bone(&mut self) {
        self.checkpoint();
        if let Some(a) = self.active_armature_mut() {
            let parent = a.active;
            a.add_bone(parent);
        }
        self.touch();
    }

    pub fn armature_delete_bones(&mut self) {
        let Some(armature_id) = self.active else { return };
        self.checkpoint();
        let remap = if let Some(a) = self.active_armature_mut() {
            a.delete_selected()
        } else { return };

        // Keep every bound mesh's joint indices valid after bone indices shift.
        for object in &mut self.scene.objects {
            let Some(skin) = object.skin.as_mut() else { continue };
            if skin.armature_id != armature_id { continue; }
            for vertex in 0..skin.joints.len().min(skin.weights.len()) {
                let mut joints = skin.joints[vertex];
                let mut weights = skin.weights[vertex];
                for slot in 0..4 {
                    let old = joints[slot] as usize;
                    if let Some(Some(new)) = remap.get(old) {
                        joints[slot] = *new as u16;
                    } else {
                        weights[slot] = 0.0;
                    }
                }
                let sum: f32 = weights.iter().sum();
                if sum > 1.0e-8 {
                    for weight in &mut weights { *weight /= sum; }
                } else {
                    // Keep a deterministic fallback binding. With zero bones the
                    // skin evaluator returns rest positions; a later first bone
                    // can safely become joint 0 instead of collapsing the mesh.
                    joints = [0; 4];
                    weights = [1.0, 0.0, 0.0, 0.0];
                }
                skin.joints[vertex] = joints;
                skin.weights[vertex] = weights;
            }
        }
        self.apply_all_skinning();
        self.touch();
    }

    pub fn armature_clear_pose(&mut self) {
        self.checkpoint();
        if let Some(a) = self.active_armature_mut() {
            a.clear_pose();
        }
        self.apply_all_skinning();
        self.touch();
    }

    /// Bind every selected mesh to the active armature (auto weights).
    pub fn bind_selected_to_armature(&mut self) -> Result<usize, String> {
        let arm_id = self.active.ok_or_else(|| "Select an Armature first".to_string())?;
        let arm_idx = self.scene.index_of(arm_id).ok_or_else(|| "Armature not found".to_string())?;
        let arm = match &self.scene.objects[arm_idx].kind {
            ObjectKind::Armature(a) => a.clone(),
            _ => return Err("Active object is not an Armature".into()),
        };
        if arm.bones.is_empty() {
            return Err("Armature has no bones".into());
        }
        let mesh_ids: Vec<u64> = self
            .selection
            .iter()
            .copied()
            .filter(|&id| id != arm_id)
            .filter(|&id| {
                self.scene
                    .objects
                    .iter()
                    .any(|o| o.id == id && o.kind.is_mesh())
            })
            .collect();
        if mesh_ids.is_empty() {
            return Err("Also select one or more Mesh objects to bind".into());
        }
        self.checkpoint();
        let worlds = self.scene.world_matrices();
        let arm_world = worlds[arm_idx];
        let mut count = 0;
        for mid in mesh_ids {
            let Some(mi) = self.scene.index_of(mid) else { continue };
            let mesh_world = worlds[mi];
            // Bones into mesh-local space for weighting
            let to_mesh = mesh_world.inverse() * arm_world;
            let mut arm_local = arm.clone();
            for b in &mut arm_local.bones {
                let h = to_mesh.transform_point3(glam::Vec3::from(b.head));
                let t = to_mesh.transform_point3(glam::Vec3::from(b.tail));
                b.head = h.to_array();
                b.tail = t.to_array();
            }
            let mesh = self.scene.objects[mi].mesh.clone();
            let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();
            let normals: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.normal).collect();
            let skin = SkinBinding::auto_weights(arm_id, &arm_local, &positions, &normals);
            // Remember the mesh->armature transform used at bind time for deform
            self.scene.objects[mi].skin = Some(skin);
            // Parent under armature (optional hierarchy)
            self.scene.objects[mi].parent = Some(arm_id);
            let local = arm_world.inverse() * mesh_world;
            let (s, r, t) = local.to_scale_rotation_translation();
            self.scene.objects[mi].scale = s;
            self.scene.objects[mi].rotation = r;
            self.scene.objects[mi].translation = t;
            count += 1;
        }
        self.apply_all_skinning();
        self.touch();
        Ok(count)
    }

    /// Recompute every skinned mesh from rest pose + current armature pose.
    pub fn apply_all_skinning(&mut self) {
        // Armature world + deform (armature local)
        let mut arm_info: Vec<(u64, glam::Mat4, Vec<glam::Mat4>)> = Vec::new();
        {
            let worlds = self.scene.world_matrices();
            for (i, o) in self.scene.objects.iter().enumerate() {
                if let ObjectKind::Armature(a) = &o.kind {
                    arm_info.push((o.id, worlds[i], a.deform_matrices_hierarchical()));
                }
            }
        }
        let worlds = self.scene.world_matrices();
        for (oi, o) in self.scene.objects.iter_mut().enumerate() {
            let Some(skin) = o.skin.clone() else { continue };
            let Some((_, arm_world, deform_arm)) = arm_info.iter().find(|(id, _, _)| *id == skin.armature_id) else {
                continue;
            };
            // Mesh local deform mats: convert armature-local deform into mesh-local
            // rest was weighted in mesh-local with bones mapped to mesh at bind.
            // Pose deform in armature space D_arm; mesh-local D = inv(mesh_world) * arm_world * D_arm * inv(arm_world) * mesh_world
            // At bind, bones were mapped with to_mesh = inv(mesh_w)*arm_w. Rest is mesh local.
            // Approximate: D_mesh ≈ inv(local) * D_arm * local when parented (mesh_world = arm_world * local)
            let mesh_world = worlds[oi];
            let to_mesh = mesh_world.inverse() * *arm_world;
            let from_mesh = to_mesh.inverse();
            let deform_mesh: Vec<glam::Mat4> = deform_arm
                .iter()
                .map(|d| to_mesh * *d * from_mesh)
                .collect();
            let (pos, nrm) = skin.deform(&deform_mesh);
            let mut mesh = (*o.mesh).clone();
            for (i, v) in mesh.vertices.iter_mut().enumerate() {
                if i < pos.len() {
                    v.position = pos[i];
                }
                if i < nrm.len() {
                    v.normal = nrm[i];
                }
            }
            o.mesh = std::sync::Arc::new(mesh);
            o.primitive = None;
        }
        self.version = self.version.wrapping_add(1);
    }


}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undo_redo_add_delete() {
        let mut d = Document::new();
        assert_eq!(d.scene.objects.len(), 3); // Cube + Sun + Camera
        d.add_primitive(Primitive::Sphere { radius: 0.5, segments: 8, rings: 4 });
        assert_eq!(d.scene.objects.len(), 4);
        d.delete_selected();
        assert_eq!(d.scene.objects.len(), 3);
        assert!(d.undo());
        assert_eq!(d.scene.objects.len(), 4);
        assert!(d.redo());
        assert_eq!(d.scene.objects.len(), 3);
    }

    #[test]
    fn unique_names() {
        let mut d = Document::new();
        d.add_primitive(Primitive::Cube { size: 1.0 });
        assert_eq!(d.scene.objects[3].name, "Cube.001");
    }

    #[test]
    fn parent_keeps_world_position() {
        let mut d = Document::new();
        // Use the default cube (index 0) as parent and add a child
        d.add_primitive(Primitive::Cube { size: 1.0 });
        d.scene.objects[0].translation = Vec3::new(5.0, 0.0, 0.0);
        d.scene.objects[3].translation = Vec3::new(0.0, 3.0, 0.0);
        let child = d.scene.objects[3].id;
        d.select_only(3);
        let parent = d.scene.objects[0].id;
        d.selection.insert(parent);
        d.active = Some(parent);
        d.parent_set().unwrap();
        let w = d.scene.world_matrices();
        let ci = d.scene.index_of(child).unwrap();
        assert!((w[ci].w_axis.y - 3.0).abs() < 1e-4 && w[ci].w_axis.x.abs() < 1e-4);
    }

    #[test]
    fn join_and_separate_roundtrip() {
        let mut d = Document::new();
        d.add_primitive(Primitive::Sphere { radius: 0.3, segments: 8, rings: 4 });
        d.scene.objects[3].translation = Vec3::new(3.0, 0.0, 0.0);
        // Select only the two meshes
        d.selection.clear();
        d.selection.insert(d.scene.objects[0].id);
        d.selection.insert(d.scene.objects[3].id);
        d.active = Some(d.scene.objects[0].id);
        d.join().unwrap();
        // After join: original non-mesh objects remain + 1 joined mesh
        assert_eq!(d.scene.objects.len(), 3);
        d.select_all();
        assert_eq!(d.separate_loose(), 1);
        assert_eq!(d.scene.objects.len(), 4);
    }
}
