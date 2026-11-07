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

    fn accessor(&mut self, view: usize, offset: usize, comp: u32, count: usize, ty: &str, minmax: Option<(Vec<f32>, Vec<f32>)>) -> usize {
        let mut a = json!({ "bufferView": view, "componentType": comp, "count": count, "type": ty });
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

    let mat_ok = |m: Option<usize>| m.filter(|&i| i < nmat);
    let mut prims = Vec::new();
    let mut push = |b: &mut Builder, first: usize, count: usize, mat: Option<usize>| {
        let ia = b.accessor(iv, first * 4, 5125, count, "SCALAR", None);
        let mut prim = json!({ "attributes": { "POSITION": pa, "NORMAL": na, "TEXCOORD_0": ta }, "indices": ia, "mode": 4 });
        if let Some(m) = mat_ok(mat) {
            prim["material"] = json!(m);
        }
        prims.push(prim);
    };
    if o.mesh.groups.is_empty() {
        push(b, 0, o.mesh.indices.len(), o.material);
    } else {
        for g in &o.mesh.groups {
            push(b, g.first_index as usize, g.index_count as usize, g.material);
        }
    }
    json!({ "name": o.mesh.name, "primitives": prims })
}

/// `only`: if Some, only objects whose id is in the set are exported.
#[cfg(not(target_arch = "wasm32"))]
pub fn export_glb(scene: &Scene, path: &Path, only: Option<&BTreeSet<u64>>) -> Result<()> {
    let mut b = Builder { bin: Vec::new(), views: Vec::new(), accessors: Vec::new() };

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
        if o.kind.is_mesh() && o.mesh.indices.is_empty() {
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
        let parent_i = o.parent.and_then(|p| scene.index_of(p)).filter(|&pi| pi != i && node_of[pi].is_some());
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
                    let outer = (l.spot_angle_deg * 0.5).to_radians().clamp(0.01, std::f32::consts::FRAC_PI_2);
                    lj["spot"] = json!({ "innerConeAngle": outer * (1.0 - l.spot_blend.clamp(0.0, 0.99)), "outerConeAngle": outer });
                }
                lights.push(lj);
                node["extensions"] = json!({ "KHR_lights_punctual": { "light": lights.len() - 1 } });
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
            // Surface, Volume, GreasePencil, Armature, LightProbe, Speaker, ForceField: empty nodes
            _ => {}
        }
        nodes.push(node);
        match parent_i.and_then(|pi| node_of[pi]) {
            Some(pn) => children[pn].push(ni),
            None => roots.push(ni),
        }
    }
    for (ni, ch) in children.iter().enumerate() {
        if !ch.is_empty() {
            nodes[ni]["children"] = json!(ch);
        }
    }

    let mut root = Map::new();
    root.insert("asset".into(), json!({ "version": "2.0", "generator": "Forge3D" }));
    root.insert("scene".into(), json!(0));
    root.insert("scenes".into(), json!([{ "nodes": roots }]));
    root.insert("nodes".into(), Value::Array(nodes));
    if !meshes.is_empty() {
        root.insert("meshes".into(), Value::Array(meshes));
    }
    if !cameras.is_empty() {
        root.insert("cameras".into(), Value::Array(cameras));
    }
    if !lights.is_empty() {
        root.insert("extensionsUsed".into(), json!(["KHR_lights_punctual"]));
        root.insert("extensions".into(), json!({ "KHR_lights_punctual": { "lights": lights } }));
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
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Wavefront OBJ (geometry only, transforms baked, Y-up). Materials are not written.
#[cfg(not(target_arch = "wasm32"))]
pub fn export_obj(scene: &Scene, path: &Path, only: Option<&BTreeSet<u64>>) -> Result<()> {
    let worlds = scene.world_matrices();
    let mut s = String::from("# Exported by Forge3D\n");
    let mut off = 1usize;
    for (i, o) in scene.objects.iter().enumerate() {
        if !scene.is_visible(i) || o.mesh.indices.is_empty() || !o.kind.is_mesh() || only.map_or(false, |s| !s.contains(&o.id)) {
            continue;
        }
        let m = worlds[i];
        let nm = Mat3::from_mat4(m).inverse().transpose();
        let flip = Mat3::from_mat4(m).determinant() < 0.0;
        let name: String = o.name.chars().map(|c| if c.is_whitespace() { '_' } else { c }).collect();
        writeln!(s, "o {name}")?;
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
        for t in o.mesh.indices.chunks_exact(3) {
            let (a, b, c) = if flip { (t[0], t[2], t[1]) } else { (t[0], t[1], t[2]) };
            let (a, b, c) = (a as usize + off, b as usize + off, c as usize + off);
            writeln!(s, "f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")?;
        }
        off += o.mesh.vertices.len();
    }
    std::fs::write(path, s).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub fn export_glb(_scene: &Scene, _path: &Path, _only: Option<&BTreeSet<u64>>) -> Result<()> {
    anyhow::bail!("GLB export is not available on WebAssembly")
}
#[cfg(target_arch = "wasm32")]
pub fn export_obj(_scene: &Scene, _path: &Path, _only: Option<&BTreeSet<u64>>) -> Result<()> {
    anyhow::bail!("OBJ export is not available on WebAssembly")
}
