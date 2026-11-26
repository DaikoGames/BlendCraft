//! Export: binary glTF (.glb, with hierarchy, materials, textures, cameras, lights) and Wavefront OBJ.
use anyhow::{bail, Context, Result};
use forge_core::{AlphaMode, LightType, Object, ObjectKind, Scene};
use glam::{Mat3, Vec3};
use image::ImageEncoder;
use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

struct Builder {
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl Builder {
    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let off = self.bin.len();
        self.bin.extend_from_slice(bytes);
        let mut v = json!({ "buffer": 0, "byteOffset": off, "byteLength": bytes.len() });
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        self.views.push(v);
        self.views.len() - 1
    }

    fn accessor(
        &mut self,
        view: usize,
        offset: usize,
        comp: u32,
        count: usize,
        ty: &str,
        minmax: Option<(Vec<f32>, Vec<f32>)>,
    ) -> usize {
        let mut a =
            json!({ "bufferView": view, "componentType": comp, "count": count, "type": ty });
        if offset > 0 {
            a["byteOffset"] = json!(offset);
        }
        if let Some((mn, mx)) = minmax {
            a["min"] = json!(mn);
            a["max"] = json!(mx);
        }
        self.accessors.push(a);
        self.accessors.len() - 1
    }
}

/// Vertex data + one primitive per material group.
/// When `skin` is set, JOINTS_0 / WEIGHTS_0 attributes are written from the binding.
fn add_mesh(b: &mut Builder, o: &Object, nmat: usize) -> Value {
    let verts = &o.mesh.vertices;
    let pos: Vec<[f32; 3]> = verts.iter().map(|v| v.position).collect();
    let nor: Vec<[f32; 3]> = verts.iter().map(|v| v.normal).collect();
    let uv: Vec<[f32; 2]> = verts.iter().map(|v| v.uv).collect();
    let mut mn = vec![f32::MAX; 3];
    let mut mx = vec![f32::MIN; 3];
    for p in &pos {
        for k in 0..3 {
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
    }
    let pv = b.view(bytemuck::cast_slice(&pos), Some(34962));
    let pa = b.accessor(pv, 0, 5126, pos.len(), "VEC3", Some((mn, mx)));
    let nv = b.view(bytemuck::cast_slice(&nor), Some(34962));
    let na = b.accessor(nv, 0, 5126, nor.len(), "VEC3", None);
    let tv = b.view(bytemuck::cast_slice(&uv), Some(34962));
    let ta = b.accessor(tv, 0, 5126, uv.len(), "VEC2", None);
    let iv = b.view(bytemuck::cast_slice(&o.mesh.indices), Some(34963));

    // Skin attributes (optional).
    let mut joints_acc: Option<usize> = None;
    let mut weights_acc: Option<usize> = None;
    if let Some(skin) = o.skin.as_ref() {
        if skin.joints.len() == verts.len() && skin.weights.len() == verts.len() {
            // JOINTS as UNSIGNED_SHORT VEC4
            let mut jbytes: Vec<u8> = Vec::with_capacity(verts.len() * 8);
            for j in &skin.joints {
                for &c in j {
                    jbytes.extend_from_slice(&c.to_le_bytes());
                }
            }
            let jv = b.view(&jbytes, Some(34962));
            joints_acc = Some(b.accessor(jv, 0, 5123, verts.len(), "VEC4", None));
            // WEIGHTS as FLOAT VEC4
            let wflat: Vec<[f32; 4]> = skin.weights.clone();
            let wv = b.view(bytemuck::cast_slice(&wflat), Some(34962));
            weights_acc = Some(b.accessor(wv, 0, 5126, verts.len(), "VEC4", None));
        }
    }

    let mat_ok = |m: Option<usize>| m.filter(|&i| i < nmat);
    let mut prims = Vec::new();
    let mut push = |b: &mut Builder, first: usize, count: usize, mat: Option<usize>| {
        let ia = b.accessor(iv, first * 4, 5125, count, "SCALAR", None);
        let mut attrs = json!({ "POSITION": pa, "NORMAL": na, "TEXCOORD_0": ta });
        if let Some(ja) = joints_acc {
            attrs["JOINTS_0"] = json!(ja);
        }
        if let Some(wa) = weights_acc {
            attrs["WEIGHTS_0"] = json!(wa);
        }
        let mut prim = json!({ "attributes": attrs, "indices": ia, "mode": 4 });
        if let Some(m) = mat_ok(mat) {
            prim["material"] = json!(m);
        }
        prims.push(prim);
    };
    if o.mesh.groups.is_empty() {
        push(b, 0, o.mesh.indices.len(), o.material);
    } else {
        for g in &o.mesh.groups {
            push(
                b,
                g.first_index as usize,
                g.index_count as usize,
                g.material,
            );
        }
    }
    json!({ "name": o.mesh.name, "primitives": prims })
}

/// `only`: if Some, only objects whose id is in the set are exported.
pub fn export_glb_to_bytes(scene: &Scene, only: Option<&BTreeSet<u64>>) -> Result<Vec<u8>> {
    let mut b = Builder {
        bin: Vec::new(),
        views: Vec::new(),
        accessors: Vec::new(),
    };

    // textures -> PNG images
    let mut images = Vec::new();
    let mut textures = Vec::new();
    for (i, t) in scene.textures.iter().enumerate() {
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&t.rgba8, t.width, t.height, image::ExtendedColorType::Rgba8)
            .context("encoding texture")?;
        let v = b.view(&png, None);
        images.push(json!({ "bufferView": v, "mimeType": "image/png", "name": t.name }));
        textures.push(json!({ "source": i }));
    }
    let ntex = textures.len();
    let tex_ok = |t: Option<usize>| t.filter(|&i| i < ntex);

    // materials
    let materials: Vec<Value> = scene
        .materials
        .iter()
        .map(|m| {
            let mut pbr = json!({ "baseColorFactor": m.base_color, "metallicFactor": m.metallic, "roughnessFactor": m.roughness });
            if let Some(t) = tex_ok(m.base_color_tex) {
                pbr["baseColorTexture"] = json!({ "index": t });
            }
            if let Some(t) = tex_ok(m.metallic_roughness_tex) {
                pbr["metallicRoughnessTexture"] = json!({ "index": t });
            }
            let mode = match m.alpha_mode {
                AlphaMode::Opaque => "OPAQUE",
                AlphaMode::Mask => "MASK",
                AlphaMode::Blend => "BLEND",
            };
            let mut j = json!({ "name": m.name, "pbrMetallicRoughness": pbr, "emissiveFactor": m.emissive, "alphaMode": mode, "doubleSided": m.double_sided });
            if m.alpha_mode == AlphaMode::Mask {
                j["alphaCutoff"] = json!(m.alpha_cutoff);
            }
            if let Some(t) = tex_ok(m.normal_tex) {
                j["normalTexture"] = json!({ "index": t, "scale": m.normal_scale });
            }
            if let Some(t) = tex_ok(m.occlusion_tex) {
                j["occlusionTexture"] = json!({ "index": t, "strength": m.occlusion_strength });
            }
            if let Some(t) = tex_ok(m.emissive_tex) {
                j["emissiveTexture"] = json!({ "index": t });
            }
            j
        })
        .collect();

    // which objects become nodes
    let worlds = scene.world_matrices();
    let n = scene.objects.len();
    let mut node_of: Vec<Option<usize>> = vec![None; n];
    let mut order: Vec<usize> = Vec::new();
    for i in 0..n {
        if !scene.is_visible(i) {
            continue;
        }
        let o = &scene.objects[i];
        if only.map_or(false, |s| !s.contains(&o.id)) {
            continue;
        }
        // Keep armatures (bones) even without mesh geometry; skip empty mesh-only objects.
        let is_arm = matches!(o.kind, ObjectKind::Armature(_));
        if o.kind.is_mesh() && o.mesh.indices.is_empty() && !is_arm {
            continue;
        }
        node_of[i] = Some(order.len());
        order.push(i);
    }
    if order.is_empty() {
        bail!("nothing to export (no visible or selected objects)");
    }

    let mut mesh_ids: HashMap<(usize, Option<usize>), usize> = HashMap::new();
    let mut meshes: Vec<Value> = Vec::new();
    let mut cameras: Vec<Value> = Vec::new();
    let mut lights: Vec<Value> = Vec::new();
    let mut nodes: Vec<Value> = Vec::new();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); order.len()];
    let mut roots: Vec<usize> = Vec::new();

    for (ni, &i) in order.iter().enumerate() {
        let o = &scene.objects[i];
        // parent node (if the parent is exported) -> local transform relative to it; else world transform
        let parent_i = o
            .parent
            .and_then(|p| scene.index_of(p))
            .filter(|&pi| pi != i && node_of[pi].is_some());
        let local = match parent_i {
            Some(pi) if worlds[pi].determinant().abs() > 1e-12 => worlds[pi].inverse() * worlds[i],
            _ => worlds[i],
        };
        let (mut s, r, t) = local.to_scale_rotation_translation();
        if matches!(o.kind, ObjectKind::Camera(_) | ObjectKind::Light(_)) {
            s = Vec3::ONE;
        }
        let mut node = json!({ "name": o.name, "translation": t.to_array(), "rotation": r.to_array(), "scale": s.to_array() });
        match &o.kind {
            ObjectKind::Empty { .. } => {}
            ObjectKind::Camera(c) => {
                let cam = if c.ortho {
                    let ym = c.ortho_size * 0.5;
                    json!({ "type": "orthographic", "orthographic": { "xmag": ym * 16.0 / 9.0, "ymag": ym, "znear": c.near, "zfar": c.far } })
                } else {
                    json!({ "type": "perspective", "perspective": { "yfov": c.fov_deg.to_radians(), "znear": c.near, "zfar": c.far } })
                };
                cameras.push(cam);
                node["camera"] = json!(cameras.len() - 1);
            }
            ObjectKind::Light(l) => {
                let ty = match l.kind {
                    LightType::Point | LightType::Area => "point",
                    LightType::Sun => "directional",
                    LightType::Spot => "spot",
                };
                let mut lj = json!({ "name": o.name, "type": ty, "color": l.color, "intensity": l.intensity });
                if l.range > 0.0 && l.kind != LightType::Sun {
                    lj["range"] = json!(l.range);
                }
                if l.kind == LightType::Spot {
                    let outer = (l.spot_angle_deg * 0.5)
                        .to_radians()
                        .clamp(0.01, std::f32::consts::FRAC_PI_2);
                    lj["spot"] = json!({ "innerConeAngle": outer * (1.0 - l.spot_blend.clamp(0.0, 0.99)), "outerConeAngle": outer });
                }
                lights.push(lj);
                node["extensions"] =
                    json!({ "KHR_lights_punctual": { "light": lights.len() - 1 } });
            }
            ObjectKind::Mesh => {
                let mat = o.material.filter(|&m| m < scene.materials.len());
                let key = (Arc::as_ptr(&o.mesh) as usize, mat);
                let mi = match mesh_ids.get(&key) {
                    Some(&m) => m,
                    None => {
                        let mj = add_mesh(&mut b, o, scene.materials.len());
                        meshes.push(mj);
                        let id = meshes.len() - 1;
                        mesh_ids.insert(key, id);
                        id
                    }
                };
                node["mesh"] = json!(mi);
            }
            ObjectKind::Armature(_) => {
                // Joint nodes are appended below.
            }
            // Surface, Volume, GreasePencil, LightProbe, Speaker, ForceField: empty nodes
            _ => {}
        }
        nodes.push(node);
        match parent_i.and_then(|pi| node_of[pi]) {
            Some(pn) => children[pn].push(ni),
            None => roots.push(ni),
        }
    }

    // ---- Skins / bones ------------------------------------------------------------
    // Joint nodes live under each armature. Skinned meshes get JOINTS/WEIGHTS via
    // add_mesh and a `skin` index pointing at skins_json.
    let mut skins_json: Vec<Value> = Vec::new();
    let mut skin_index_of: HashMap<u64, usize> = HashMap::new();
    for (ni, &i) in order.iter().enumerate() {
        let o = &scene.objects[i];
        let ObjectKind::Armature(arm) = &o.kind else {
            continue;
        };
        if arm.bones.is_empty() {
            continue;
        }
        let arm_world = worlds[i];
        let rest_mats: Vec<glam::Mat4> = arm
            .bones
            .iter()
            .map(forge_core::ArmatureData::rest_matrix)
            .collect();
        let mut joint_node_indices: Vec<usize> = Vec::with_capacity(arm.bones.len());
        let mut bone_children: Vec<Vec<usize>> = vec![Vec::new(); arm.bones.len()];
        for (bi, bone) in arm.bones.iter().enumerate() {
            let rest = rest_mats[bi];
            let local = if let Some(p) = bone.parent {
                let pi = p as usize;
                if pi < rest_mats.len() && rest_mats[pi].determinant().abs() > 1e-12 {
                    rest_mats[pi].inverse() * rest
                } else {
                    rest
                }
            } else {
                rest
            };
            let (s, r, t) = local.to_scale_rotation_translation();
            let jni = nodes.len();
            nodes.push(json!({
                "name": bone.name,
                "translation": t.to_array(),
                "rotation": r.to_array(),
                "scale": s.to_array()
            }));
            joint_node_indices.push(jni);
            if let Some(p) = bone.parent {
                let pi = p as usize;
                if pi < bone_children.len() {
                    bone_children[pi].push(jni);
                }
            } else {
                children[ni].push(jni);
            }
        }
        for (bi, ch) in bone_children.iter().enumerate() {
            if !ch.is_empty() {
                nodes[joint_node_indices[bi]]["children"] = json!(ch);
            }
        }
        let mut ibm: Vec<[f32; 16]> = Vec::with_capacity(arm.bones.len());
        for rest in &rest_mats {
            let global = arm_world * *rest;
            ibm.push(global.inverse().to_cols_array());
        }
        let ibm_view = b.view(bytemuck::cast_slice(&ibm), None);
        let ibm_acc = b.accessor(ibm_view, 0, 5126, ibm.len(), "MAT4", None);
        let si = skins_json.len();
        skins_json.push(json!({
            "name": o.name,
            "joints": joint_node_indices,
            "inverseBindMatrices": ibm_acc,
            "skeleton": ni
        }));
        skin_index_of.insert(o.id, si);
    }
    // Link skinned meshes to their skin.
    for (ni, &i) in order.iter().enumerate() {
        if let Some(skin) = scene.objects[i].skin.as_ref() {
            if let Some(&si) = skin_index_of.get(&skin.armature_id) {
                nodes[ni]["skin"] = json!(si);
            }
        }
    }

    for (ni, ch) in children.iter().enumerate() {
        if !ch.is_empty() {
            nodes[ni]["children"] = json!(ch);
        }
    }

    let mut root = Map::new();
    root.insert(
        "asset".into(),
        json!({ "version": "2.0", "generator": "BlendCraft" }),
    );
    root.insert("scene".into(), json!(0));
    root.insert("scenes".into(), json!([{ "nodes": roots }]));
    root.insert("nodes".into(), Value::Array(nodes));
    if !meshes.is_empty() {
        root.insert("meshes".into(), Value::Array(meshes));
    }
    if !skins_json.is_empty() {
        root.insert("skins".into(), Value::Array(skins_json));
    }
    if !cameras.is_empty() {
        root.insert("cameras".into(), Value::Array(cameras));
    }
    if !lights.is_empty() {
        root.insert("extensionsUsed".into(), json!(["KHR_lights_punctual"]));
        root.insert(
            "extensions".into(),
            json!({ "KHR_lights_punctual": { "lights": lights } }),
        );
    }
    if !materials.is_empty() {
        root.insert("materials".into(), Value::Array(materials));
    }
    if !textures.is_empty() {
        root.insert("textures".into(), Value::Array(textures));
        root.insert("images".into(), Value::Array(images));
    }
    while b.bin.len() % 4 != 0 {
        b.bin.push(0);
    }
    if !b.bin.is_empty() {
        root.insert("accessors".into(), Value::Array(b.accessors));
        root.insert("bufferViews".into(), Value::Array(b.views));
        root.insert("buffers".into(), json!([{ "byteLength": b.bin.len() }]));
    }

    let mut json_bytes = serde_json::to_vec(&Value::Object(root))?;
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    let total = 12 + 8 + json_bytes.len() + 8 + b.bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&(b.bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004E4942u32.to_le_bytes());
    out.extend_from_slice(&b.bin);
    Ok(out)
}

/// Write a GLB file to disk.
#[cfg(not(target_arch = "wasm32"))]
pub fn export_glb(scene: &Scene, path: &Path, only: Option<&BTreeSet<u64>>) -> Result<()> {
    let out = export_glb_to_bytes(scene, only)?;
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Wavefront OBJ + MTL (transforms baked, Y-up). Writes materials and PNG textures
/// next to the OBJ (same stem `.mtl`, textures as `{stem}_{texname}.png`).
#[cfg(not(target_arch = "wasm32"))]
pub fn export_obj(scene: &Scene, path: &Path, only: Option<&BTreeSet<u64>>) -> Result<()> {
    let worlds = scene.world_matrices();
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("export");
    let mtl_name = format!("{stem}.mtl");
    let mtl_path = dir.join(&mtl_name);

    // Collect materials actually used by exported meshes.
    let mut used_mats: BTreeSet<usize> = BTreeSet::new();
    for (i, o) in scene.objects.iter().enumerate() {
        if !scene.is_visible(i)
            || o.mesh.indices.is_empty()
            || !o.kind.is_mesh()
            || only.map_or(false, |s| !s.contains(&o.id))
        {
            continue;
        }
        if let Some(mi) = o.material {
            if mi < scene.materials.len() {
                used_mats.insert(mi);
            }
        }
        for g in &o.mesh.groups {
            if let Some(mi) = g.material {
                if mi < scene.materials.len() {
                    used_mats.insert(mi);
                }
            }
        }
    }

    // Write texture PNGs referenced by used materials; map scene tex index -> file name.
    let mut tex_files: HashMap<usize, String> = HashMap::new();
    for &mi in &used_mats {
        let m = &scene.materials[mi];
        for ti in [m.base_color_tex, m.normal_tex, m.emissive_tex, m.occlusion_tex]
            .into_iter()
            .flatten()
        {
            if tex_files.contains_key(&ti) {
                continue;
            }
            let Some(tex) = scene.textures.get(ti) else {
                continue;
            };
            let safe: String = tex
                .name
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            let safe = if safe.is_empty() {
                format!("tex{ti}")
            } else {
                safe
            };
            let fname = format!("{stem}_{safe}.png");
            let fpath = dir.join(&fname);
            image::save_buffer(
                &fpath,
                &tex.rgba8,
                tex.width,
                tex.height,
                image::ColorType::Rgba8,
            )
            .with_context(|| format!("writing texture {}", fpath.display()))?;
            tex_files.insert(ti, fname);
        }
    }

    // MTL
    let mut mtl = String::from("# Exported by BlendCraft\n");
    let mut mat_name_of: HashMap<usize, String> = HashMap::new();
    for &mi in &used_mats {
        let m = &scene.materials[mi];
        let safe: String = m
            .name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let mname = if safe.is_empty() {
            format!("Material_{mi}")
        } else {
            format!("{safe}_{mi}")
        };
        mat_name_of.insert(mi, mname.clone());
        writeln!(mtl, "newmtl {mname}")?;
        let c = m.base_color;
        writeln!(mtl, "Kd {:.6} {:.6} {:.6}", c[0], c[1], c[2])?;
        writeln!(mtl, "d {:.6}", c[3])?;
        writeln!(mtl, "Ns {:.6}", (1.0 - m.roughness) * 200.0)?;
        writeln!(mtl, "illum 2")?;
        if let Some(ti) = m.base_color_tex {
            if let Some(f) = tex_files.get(&ti) {
                writeln!(mtl, "map_Kd {f}")?;
            }
        }
        if let Some(ti) = m.normal_tex {
            if let Some(f) = tex_files.get(&ti) {
                writeln!(mtl, "map_Bump {f}")?;
            }
        }
        writeln!(mtl)?;
    }
    std::fs::write(&mtl_path, mtl).with_context(|| format!("writing {}", mtl_path.display()))?;

    // OBJ
    let mut s = String::from("# Exported by BlendCraft\n");
    writeln!(s, "mtllib {mtl_name}")?;
    let mut off = 1usize;
    for (i, o) in scene.objects.iter().enumerate() {
        if !scene.is_visible(i)
            || o.mesh.indices.is_empty()
            || !o.kind.is_mesh()
            || only.map_or(false, |s| !s.contains(&o.id))
        {
            continue;
        }
        let m = worlds[i];
        let nm = Mat3::from_mat4(m).inverse().transpose();
        let flip = Mat3::from_mat4(m).determinant() < 0.0;
        let name: String = o
            .name
            .chars()
            .map(|c| if c.is_whitespace() { '_' } else { c })
            .collect();
        writeln!(s, "o {name}")?;
        if let Some(mi) = o.material {
            if let Some(mn) = mat_name_of.get(&mi) {
                writeln!(s, "usemtl {mn}")?;
            }
        }
        for v in &o.mesh.vertices {
            let p = m.transform_point3(v.position.into());
            writeln!(s, "v {} {} {}", p.x, p.y, p.z)?;
        }
        for v in &o.mesh.vertices {
            writeln!(s, "vt {} {}", v.uv[0], 1.0 - v.uv[1])?;
        }
        for v in &o.mesh.vertices {
            let n = (nm * Vec3::from(v.normal)).normalize_or_zero();
            writeln!(s, "vn {} {} {}", n.x, n.y, n.z)?;
        }
        // Per-group materials when present.
        if o.mesh.groups.is_empty() {
            for t in o.mesh.indices.chunks_exact(3) {
                let (a, b, c) = if flip {
                    (t[0], t[2], t[1])
                } else {
                    (t[0], t[1], t[2])
                };
                let (a, b, c) = (a as usize + off, b as usize + off, c as usize + off);
                writeln!(s, "f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")?;
            }
        } else {
            for g in &o.mesh.groups {
                if let Some(mi) = g.material {
                    if let Some(mn) = mat_name_of.get(&mi) {
                        writeln!(s, "usemtl {mn}")?;
                    }
                }
                let start = g.first_index as usize;
                let end = start + g.index_count as usize;
                let indices = &o.mesh.indices[start.min(o.mesh.indices.len())..end.min(o.mesh.indices.len())];
                for t in indices.chunks_exact(3) {
                    let (a, b, c) = if flip {
                        (t[0], t[2], t[1])
                    } else {
                        (t[0], t[1], t[2])
                    };
                    let (a, b, c) = (a as usize + off, b as usize + off, c as usize + off);
                    writeln!(s, "f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")?;
                }
            }
        }
        off += o.mesh.vertices.len();
    }
    std::fs::write(path, s).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub fn export_glb(scene: &Scene, _path: &Path, only: Option<&BTreeSet<u64>>) -> Result<()> {
    // Path is unused on wasm; app layer downloads `export_glb_to_bytes` instead.
    let _ = export_glb_to_bytes(scene, only)?;
    Ok(())
}
#[cfg(target_arch = "wasm32")]
pub fn export_obj(_scene: &Scene, _path: &Path, _only: Option<&BTreeSet<u64>>) -> Result<()> {
    anyhow::bail!("OBJ export on WebAssembly uses the in-app download path")
}
