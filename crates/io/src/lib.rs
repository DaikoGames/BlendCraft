//! File import/export. glTF 2.0 / GLB (geometry, node transforms, PBR materials, textures)
//! and HDR environment maps.
use anyhow::{Context, Result};
use forge_core::{
    AlphaMode, ArmatureData, Bone, EnvironmentMap, Material, MeshData, Object, ObjectKind, Scene, TextureData, Vertex,
};
use glam::{Mat4, Vec3};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

mod alembic;
mod export;
mod fbx;
mod fbx_ufbx;
mod obj;
mod usd;
#[cfg(not(target_arch = "wasm32"))]
mod unity;
pub use alembic::{
    build_alembic_bytes, export_alembic, load_alembic, parse_alembic_bytes, AlembicImport,
    ImportedObject,
};
pub use export::{export_glb, export_obj};
pub use fbx::{export_fbx, load_fbx, load_fbx_bytes, load_fbx_bytes_with};
pub use obj::load_obj;
pub use usd::{build_usda_string, export_usda};
#[cfg(not(target_arch = "wasm32"))]
pub use unity::load_unitypackage;


/// Decode any supported image file (PNG/JPEG/TGA/BMP/...) held in memory into RGBA8.
/// `name_hint` (a file name or path) is only used to pick a decoder for formats without a
/// magic number (TGA) and to name the texture.
pub(crate) fn decode_image_bytes(bytes: &[u8], name_hint: &str) -> Option<TextureData> {
    let lower = name_hint.to_ascii_lowercase();
    let img = match image::load_from_memory(bytes) {
        Ok(i) => i,
        Err(first) => {
            let fmt = if lower.ends_with(".tga") {
                Some(image::ImageFormat::Tga)
            } else if lower.ends_with(".bmp") {
                Some(image::ImageFormat::Bmp)
            } else {
                None
            };
            match fmt.and_then(|f| image::load_from_memory_with_format(bytes, f).ok()) {
                Some(i) => i,
                None => {
                    eprintln!("texture '{name_hint}' could not be decoded: {first}");
                    return None;
                }
            }
        }
    };
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    let norm = name_hint.replace('\\', "/");
    let file = norm.rsplit('/').next().unwrap_or(&norm);
    let name = file
        .rsplit_once('.')
        .map(|(s, _)| s)
        .unwrap_or(file)
        .to_string();
    Some(TextureData {
        name: if name.is_empty() { "Texture".into() } else { name },
        width,
        height,
        rgba8: rgba.into_raw(),
    })
}

/// Look for a texture file referenced by a model. Model formats store the path of the
/// artist's machine (often absolute Windows paths), so besides the path as written we also
/// try the bare file name next to the model and in the usual texture folders.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn find_texture_file(dir: &Path, model_stem: &str, name: &str) -> Option<Vec<u8>> {
    use std::path::PathBuf;
    let norm = name.trim().replace('\\', "/");
    let file = norm.rsplit('/').next().unwrap_or(&norm).to_string();
    if file.is_empty() {
        return None;
    }
    let mut tries: Vec<PathBuf> = Vec::new();
    let p = Path::new(&norm);
    if p.is_absolute() {
        tries.push(p.to_path_buf());
    }
    tries.push(dir.join(&norm));
    tries.push(dir.join(&file));
    if !model_stem.is_empty() {
        tries.push(dir.join(format!("{model_stem}.fbm")).join(&file));
    }
    for sub in [
        "textures", "Textures", "texture", "Texture", "tex", "Tex", "maps", "Maps", "images",
        "Images", "materials", "Materials",
    ] {
        tries.push(dir.join(sub).join(&file));
        if let Some(parent) = dir.parent() {
            tries.push(parent.join(sub).join(&file));
        }
    }
    for t in tries {
        if t.is_file() {
            if let Ok(b) = std::fs::read(&t) {
                return Some(b);
            }
        }
    }
    // Last resort: case-insensitive match in the model's folder (Windows-authored files).
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().eq_ignore_ascii_case(&file) {
                if let Ok(b) = std::fs::read(e.path()) {
                    return Some(b);
                }
            }
        }
    }
    None
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_gltf(path: &Path) -> Result<Scene> {
    let (doc, buffers, images) =
        gltf::import(path).with_context(|| format!("reading {}", path.display()))?;
    let mut scene = Scene::default();
    scene.materials = doc.materials().map(|m| convert_material(&m)).collect();
    // Decode every image. glTF embeds PNG/JPEG in the GLB binary (or as URIs).
    // If the crate returns empty data (rare decode miss), fall back to decoding the
    // buffer-view bytes with the `image` crate so Cycles still gets albedo maps.
    let mut textures = Vec::with_capacity(images.len());
    for (img, data) in doc.images().zip(images.iter()) {
        let name = img.name().unwrap_or("Image");
        let mut tex = convert_image(name, data);
        if tex.width == 0 || tex.height == 0 || tex.rgba8.is_empty() {
            // Buffer-view embedded image (typical for .glb)
            match img.source() {
                gltf::image::Source::View { view, mime_type: _ } => {
                    let buf = &buffers[view.buffer().index()].0;
                    let start = view.offset();
                    let end = start + view.length();
                    if end <= buf.len() {
                        if let Some(decoded) = decode_image_bytes(&buf[start..end], name) {
                            tex = decoded;
                        }
                    }
                }
                gltf::image::Source::Uri { uri, .. } => {
                    if let Some(bytes) = find_texture_file(
                        path.parent().unwrap_or_else(|| Path::new(".")),
                        path.file_stem().and_then(|s| s.to_str()).unwrap_or(""),
                        uri,
                    ) {
                        if let Some(decoded) = decode_image_bytes(&bytes, name) {
                            tex = decoded;
                        }
                    }
                }
            }
        }
        textures.push(tex);
    }
    scene.textures = Arc::new(textures);
    // Roots of the default scene; files without any scene fall back to every parentless node.
    let roots: Vec<gltf::Node> = match doc.default_scene().or_else(|| doc.scenes().next()) {
        Some(s) => s.nodes().collect(),
        None => {
            let mut is_child = vec![false; doc.nodes().count()];
            for n in doc.nodes() {
                for c in n.children() {
                    is_child[c.index()] = true;
                }
            }
            doc.nodes().filter(|n| !is_child[n.index()]).collect()
        }
    };
    if roots.is_empty() {
        anyhow::bail!("file has no scene");
    }
    // World matrix of every node: skinned meshes are posed by their joints' world matrices.
    let mut worlds: HashMap<usize, Mat4> = HashMap::new();
    for node in &roots {
        collect_worlds(node, Mat4::IDENTITY, &mut worlds);
    }
    // Preserve glTF skins as editable BlendCraft armature objects. The previous
    // importer only baked skin deformation into vertex positions, which discarded
    // the source skeleton entirely. Joint node transforms are converted into
    // armature-local rest positions; joint-parent relationships become bones.
    // gltf::Node does not expose a parent() accessor. Build the reverse
    // hierarchy once from child links so joints can resolve their parent.
    let mut parent_by_node: HashMap<usize, usize> = HashMap::new();
    for parent in doc.nodes() {
        for child in parent.children() {
            parent_by_node.insert(child.index(), parent.index());
        }
    }
    let mut armature_ids: HashMap<usize, u64> = HashMap::new();
    let mut armature_worlds: HashMap<u64, Mat4> = HashMap::new();
    let mut node_to_bone: HashMap<usize, (u64, u32)> = HashMap::new();
    let mut node_rest_trs: HashMap<usize, (Vec3, glam::Quat, Vec3)> = HashMap::new();
    for skin in doc.skins() {
        let joints: Vec<gltf::Node> = skin.joints().collect();
        if joints.is_empty() { continue; }
        let joint_to_bone: HashMap<usize, u32> = joints.iter().enumerate()
            .map(|(i, joint)| (joint.index(), i as u32)).collect();
        // Use the glTF skeleton root as the armature object's transform and
        // store every bone in armature-local space. Previously joint world-space
        // positions were stored as bone-local coordinates while the armature
        // object stayed at identity, causing an offset/double-transform at rest.
        let armature_world = skin.skeleton()
            .and_then(|root| worlds.get(&root.index()).copied())
            .or_else(|| joints.iter().find(|joint| {
                parent_by_node.get(&joint.index())
                    .is_none_or(|parent_index| !joint_to_bone.contains_key(parent_index))
            }).and_then(|root| worlds.get(&root.index()).copied()))
            .unwrap_or(Mat4::IDENTITY);
        let world_to_armature = armature_world.inverse();

        // Rest pose always comes from joint NODE transforms so animation channels
        // (absolute local TRS on those nodes) stay in the same space.
        // Inverse-bind matrices are stored on SkinBinding and used only for LBS
        // skinning — they are NOT used to replace rest (that mixed spaces and
        // broke both T-pose display and animation).
        let mut arm = ArmatureData::default();
        arm.bones.clear();
        for joint in &joints {
            let joint_world = worlds.get(&joint.index()).copied().unwrap_or(Mat4::IDENTITY);
            let joint_local = world_to_armature * joint_world;
            let head = joint_local.transform_point3(Vec3::ZERO);
            let (_, joint_rot, _) = joint_local.to_scale_rotation_translation();
            let joint_rot = joint_rot.normalize();
            let local_y = (joint_rot * Vec3::Y).normalize_or_zero();
            let mut tail = if local_y.length_squared() > 1e-8 {
                head + local_y * 0.1
            } else {
                head + Vec3::Y * 0.1
            };
            if let Some(child) = joint.children().find(|c| joint_to_bone.contains_key(&c.index())) {
                let child_world = worlds.get(&child.index()).copied().unwrap_or(Mat4::IDENTITY);
                let child_head = (world_to_armature * child_world).transform_point3(Vec3::ZERO);
                if (child_head - head).length_squared() > 1e-10 {
                    tail = child_head;
                }
            } else if let Some(parent_index) = parent_by_node.get(&joint.index()) {
                let parent_world = worlds.get(parent_index).copied().unwrap_or(Mat4::IDENTITY);
                let p = (world_to_armature * parent_world).transform_point3(Vec3::ZERO);
                let direction = (head - p).normalize_or_zero();
                if direction.length_squared() > 1e-6 && local_y.length_squared() <= 1e-8 {
                    tail = head + direction * 0.1;
                }
            }
            let parent_bone = parent_by_node.get(&joint.index())
                .and_then(|parent_index| joint_to_bone.get(parent_index).copied());
            let mut bone = Bone::new(joint.name().unwrap_or("Joint"), head, tail);
            bone.parent = parent_bone;
            bone.rest_orientation = Some(joint_rot.to_array());
            // Absolute local TRS of the joint node (same space as animation keys).
            let (lt, lr, ls) = joint.transform().decomposed();
            let lr = glam::Quat::from_array(lr).normalize();
            bone.rest_local_loc = Some(lt);
            bone.rest_local_rot = Some(lr.to_array());
            bone.rest_local_scale = Some(ls);
            bone.pose_loc = lt;
            bone.pose_rot = lr.to_array();
            bone.pose_scale = ls;
            arm.bones.push(bone);
        }
        arm.active = Some(0);
        arm.selected.clear();
        arm.selected.insert(0);
        let mut arm_obj = Object::from_matrix(
            skin.name().unwrap_or("Armature"), armature_world, Arc::new(MeshData::default())
        );
        arm_obj.id = scene.alloc_id();
        arm_obj.kind = ObjectKind::Armature(arm);
        armature_ids.insert(skin.index(), arm_obj.id);
        armature_worlds.insert(arm_obj.id, armature_world);
        for (bone_index, joint) in joints.iter().enumerate() {
            node_to_bone.insert(joint.index(), (arm_obj.id, bone_index as u32));
            let (t, r, scale) = joint.transform().decomposed();
            node_rest_trs.insert(joint.index(), (Vec3::from_array(t), glam::Quat::from_array(r).normalize(), Vec3::from_array(scale)));
        }
        scene.objects.push(arm_obj);
    }
    for node in &roots {
        visit(&node, Mat4::IDENTITY, &buffers, &worlds, &armature_ids, &armature_worlds, &mut scene);
    }
    // Import glTF animation channels into per-armature bone tracks. glTF node
    // animation targets the joint node; node_to_bone maps that node to the
    // corresponding editable BlendCraft bone.
    for (animation_index, animation) in doc.animations().enumerate() {
        let mut min_time = f32::INFINITY;
        let mut max_time = f32::NEG_INFINITY;
        let mut channel_data = Vec::new();
        for channel in animation.channels() {
            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()].0[..]));
            let Some(times) = reader.read_inputs() else { continue };
            let times: Vec<f32> = times.collect();
            if times.is_empty() { continue; }
            for &time in &times {
                min_time = min_time.min(time);
                max_time = max_time.max(time);
            }
            let Some(outputs) = reader.read_outputs() else { continue };
            channel_data.push((channel.target().node().index(), channel.target().property(), channel.sampler().interpolation(), times, outputs));
        }
        if !min_time.is_finite() { min_time = 0.0; }
        if !max_time.is_finite() { max_time = min_time; }
        let fps = scene.animation.fps.max(1) as f32;
        let frame_start = 1;
        let frame_end = ((max_time - min_time).max(0.0) * fps).ceil() as i32 + frame_start;
        scene.animation.clips.push(forge_core::AnimationClip {
            name: animation.name().map(str::to_owned).unwrap_or_else(|| format!("Animation {}", animation_index + 1)),
            frame_start,
            frame_end: frame_end.max(frame_start),
        });

        let mut clip_bone_tracks: HashMap<u64, HashMap<u32, forge_core::ObjectAnimation>> = HashMap::new();
        for (node_index, property, interpolation, times, outputs) in channel_data {
            let Some(&(armature_id, bone_index)) = node_to_bone.get(&node_index) else { continue };
            let track = clip_bone_tracks.entry(armature_id).or_default()
                .entry(bone_index).or_default();
            let to_frame = |time: f32| frame_start + ((time - min_time) * fps).round() as i32;
            match (property, outputs) {
                // Store absolute local TRS (same space as joint.transform). Hierarchical
                // eval composes parent * local; do NOT subtract rest — that mixed parent
                // local space with armature space and made motion crawl along +Y.
                (gltf::animation::Property::Translation, gltf::animation::util::ReadOutputs::Translations(values)) => {
                    let values: Vec<[f32; 3]> = values.collect();
                    let values: Vec<[f32; 3]> = if interpolation == gltf::animation::Interpolation::CubicSpline {
                        values.chunks_exact(3).map(|triple| triple[1]).collect()
                    } else { values };
                    for (time, value) in times.into_iter().zip(values) {
                        track.insert_or_merge(to_frame(time), Some(Vec3::from_array(value)), None, None);
                    }
                }
                (gltf::animation::Property::Rotation, gltf::animation::util::ReadOutputs::Rotations(values)) => {
                    let values: Vec<[f32; 4]> = values.into_f32().collect();
                    let values: Vec<[f32; 4]> = if interpolation == gltf::animation::Interpolation::CubicSpline {
                        values.chunks_exact(3).map(|triple| triple[1]).collect()
                    } else { values };
                    for (time, value) in times.into_iter().zip(values) {
                        let q = glam::Quat::from_array(value).normalize();
                        track.insert_or_merge(to_frame(time), None, Some(q), None);
                    }
                }
                (gltf::animation::Property::Scale, gltf::animation::util::ReadOutputs::Scales(values)) => {
                    let values: Vec<[f32; 3]> = values.collect();
                    let values: Vec<[f32; 3]> = if interpolation == gltf::animation::Interpolation::CubicSpline {
                        values.chunks_exact(3).map(|triple| triple[1]).collect()
                    } else { values };
                    for (time, value) in times.into_iter().zip(values) {
                        track.insert_or_merge(to_frame(time), None, None, Some(Vec3::from_array(value)));
                    }
                }
                _ => {}
            }
            track.sort();
        }
        // Keep one track collection per clip so actions never overwrite one another.
        scene.animation.bone_tracks.push(clip_bone_tracks);
    }
    if !scene.animation.clips.is_empty() { scene.animation.select_clip(Some(0)); }
    scene.fix_ids();
    Ok(scene)
}

/// Decode a PNG/JPEG into an RGBA texture (used for "Image as Plane").
#[cfg(not(target_arch = "wasm32"))]
pub fn load_image(path: &Path) -> Result<TextureData> {
    let img = image::open(path)
        .with_context(|| format!("reading {}", path.display()))?
        .to_rgba8();
    let (width, height) = img.dimensions();
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Image")
        .to_string();
    Ok(TextureData {
        name,
        width,
        height,
        rgba8: img.into_raw(),
    })
}

/// Load an equirectangular environment map (.hdr preferred; png/jpg are treated as sRGB).
#[cfg(not(target_arch = "wasm32"))]
pub fn load_environment(path: &Path) -> Result<EnvironmentMap> {
    let img = image::open(path).with_context(|| format!("reading {}", path.display()))?;
    let is_hdr = matches!(
        img,
        image::DynamicImage::ImageRgb32F(_) | image::DynamicImage::ImageRgba32F(_)
    );
    let rgb = img.to_rgb32f();
    let (width, height) = rgb.dimensions();
    let mut data = rgb.into_raw();
    if !is_hdr {
        for v in data.iter_mut() {
            *v = if *v <= 0.04045 {
                *v / 12.92
            } else {
                ((*v + 0.055) / 1.055).powf(2.4)
            };
        }
    }
    Ok(EnvironmentMap {
        width,
        height,
        rgb: data,
    })
}

fn convert_material(m: &gltf::Material) -> Material {
    let pbr = m.pbr_metallic_roughness();
    let tex = |t: Option<gltf::texture::Info>| t.map(|t| t.texture().source().index());
    Material {
        name: m.name().unwrap_or("Material").to_string(),
        base_color: pbr.base_color_factor(),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        emissive: m.emissive_factor(),
        alpha_mode: match m.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        },
        alpha_cutoff: m.alpha_cutoff().unwrap_or(0.5),
        double_sided: m.double_sided(),
        normal_scale: m.normal_texture().map_or(1.0, |t| t.scale()),
        occlusion_strength: m.occlusion_texture().map_or(1.0, |t| t.strength()),
        base_color_tex: tex(pbr.base_color_texture()),
        metallic_roughness_tex: tex(pbr.metallic_roughness_texture()),
        normal_tex: m.normal_texture().map(|t| t.texture().source().index()),
        occlusion_tex: m.occlusion_texture().map(|t| t.texture().source().index()),
        emissive_tex: tex(m.emissive_texture()),
        shader_graph: None,
        displacement_scale: 0.0,
        volume_density: 0.0,
    }
}

fn convert_image(name: &str, d: &gltf::image::Data) -> TextureData {
    use gltf::image::Format as F;
    let n = d.width as usize * d.height as usize;
    let p = &d.pixels;
    let mut out: Vec<u8> = Vec::with_capacity(n * 4);
    match d.format {
        F::R8 => {
            for i in 0..n {
                out.extend_from_slice(&[p[i], p[i], p[i], 255]);
            }
        }
        F::R8G8 => {
            for i in 0..n {
                out.extend_from_slice(&[p[i * 2], p[i * 2 + 1], 0, 255]);
            }
        }
        F::R8G8B8 => {
            for i in 0..n {
                out.extend_from_slice(&[p[i * 3], p[i * 3 + 1], p[i * 3 + 2], 255]);
            }
        }
        F::R8G8B8A8 => out.extend_from_slice(p),
        // 16-bit little endian: keep the high byte
        F::R16 => {
            for i in 0..n {
                let r = p[i * 2 + 1];
                out.extend_from_slice(&[r, r, r, 255]);
            }
        }
        F::R16G16 => {
            for i in 0..n {
                out.extend_from_slice(&[p[i * 4 + 1], p[i * 4 + 3], 0, 255]);
            }
        }
        F::R16G16B16 => {
            for i in 0..n {
                out.extend_from_slice(&[p[i * 6 + 1], p[i * 6 + 3], p[i * 6 + 5], 255]);
            }
        }
        F::R16G16B16A16 => {
            for i in 0..n {
                out.extend_from_slice(&[p[i * 8 + 1], p[i * 8 + 3], p[i * 8 + 5], p[i * 8 + 7]]);
            }
        }
        _ => {
            // unsupported pixel format: magenta so it's obvious
            for _ in 0..n {
                out.extend_from_slice(&[255, 0, 255, 255]);
            }
        }
    }
    TextureData {
        name: name.to_string(),
        width: d.width,
        height: d.height,
        rgba8: out,
    }
}

fn collect_worlds(node: &gltf::Node, parent: Mat4, out: &mut HashMap<usize, Mat4>) {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    out.insert(node.index(), world);
    for child in node.children() {
        collect_worlds(&child, world, out);
    }
}

/// Triangle list from a primitive's index list for any triangle topology.
fn triangle_indices(mode: gltf::mesh::Mode, idx: Vec<u32>) -> Option<Vec<u32>> {
    use gltf::mesh::Mode;
    match mode {
        Mode::Triangles => Some(idx),
        Mode::TriangleStrip => {
            let mut out = Vec::new();
            for i in 2..idx.len() {
                if i % 2 == 0 {
                    out.extend_from_slice(&[idx[i - 2], idx[i - 1], idx[i]]);
                } else {
                    out.extend_from_slice(&[idx[i - 1], idx[i - 2], idx[i]]);
                }
            }
            Some(out)
        }
        Mode::TriangleFan => {
            let mut out = Vec::new();
            for i in 2..idx.len() {
                out.extend_from_slice(&[idx[0], idx[i - 1], idx[i]]);
            }
            Some(out)
        }
        _ => None,
    }
}

fn visit(
    node: &gltf::Node,
    parent: Mat4,
    buffers: &[gltf::buffer::Data],
    worlds: &HashMap<usize, Mat4>,
    armature_ids: &HashMap<usize, u64>,
    armature_worlds: &HashMap<u64, Mat4>,
    scene: &mut Scene,
) {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        // Keep glTF bind-pose geometry and its original skin weights. Baking the
        // current pose here destroys editability and makes round-trip export impossible.
        let skin_armature_id = node.skin().and_then(|skin| armature_ids.get(&skin.index()).copied());
        for (i, prim) in mesh.primitives().enumerate() {
            let reader = prim.reader(|b| Some(&buffers[b.index()].0[..]));
            let Some(pos) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<[f32; 3]> = pos.collect();
            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|n| n.collect());
            // Prefer the UV set referenced by the base-color texture (often 0, sometimes 1).
            // Many GLBs put albedo on TEXCOORD_1; only reading set 0 left Cycles sampling
            // black/white while the mesh looked textured in other apps.
            let uv_set = prim
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|info| info.tex_coord())
                .unwrap_or(0);
            let uvs: Option<Vec<[f32; 2]>> = reader
                .read_tex_coords(uv_set)
                .or_else(|| reader.read_tex_coords(0))
                .map(|t| t.into_f32().collect());
            let raw_indices: Vec<u32> = match reader.read_indices() {
                Some(idx) => idx.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            let Some(indices) = triangle_indices(prim.mode(), raw_indices) else {
                continue; // points / lines
            };
            // JOINTS_n / WEIGHTS_n + inverse-bind matrices (the latter is required for
            // correct bone↔mesh alignment on real-world GLB files).
            let skin_binding = if let Some(armature_id) = skin_armature_id {
                match (reader.read_joints(0), reader.read_weights(0)) {
                    (Some(joints), Some(weights)) => {
                        let joints: Vec<[u16; 4]> = joints.into_u16().collect();
                        let weights: Vec<[f32; 4]> = weights.into_f32().collect();
                        if joints.len() == positions.len() && weights.len() == positions.len() {
                            let mut inverse_binds: Vec<[f32; 16]> = Vec::new();
                            if let Some(skin) = node.skin() {
                                let skin_reader = skin.reader(|b| Some(&buffers[b.index()].0[..]));
                                if let Some(ibms) = skin_reader.read_inverse_bind_matrices() {
                                    for m in ibms {
                                        // glTF matrices are column-major; build Mat4 then flatten
                                        // with to_cols_array so storage matches from_cols_array
                                        // used at skinning time.
                                        let mat = Mat4::from_cols(
                                            glam::Vec4::new(m[0][0], m[0][1], m[0][2], m[0][3]),
                                            glam::Vec4::new(m[1][0], m[1][1], m[1][2], m[1][3]),
                                            glam::Vec4::new(m[2][0], m[2][1], m[2][2], m[2][3]),
                                            glam::Vec4::new(m[3][0], m[3][1], m[3][2], m[3][3]),
                                        );
                                        inverse_binds.push(mat.to_cols_array());
                                    }
                                }
                            }
                            Some(forge_core::SkinBinding {
                                armature_id,
                                joints,
                                weights,
                                rest_positions: positions.clone(),
                                rest_normals: normals.clone().unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]),
                                inverse_binds,
                            })
                        } else { None }
                    }
                    _ => None,
                }
            } else { None };
            let vertices = positions
                .iter()
                .enumerate()
                .map(|(k, p)| Vertex {
                    position: *p,
                    normal: normals.as_ref().map_or([0.0; 3], |n| n[k]),
                    uv: uvs.as_ref().map_or([0.0; 2], |u| u[k]),
                    color: [1.0, 1.0, 1.0, 1.0],
                })
                .collect();
            let mut md = MeshData {
                name: mesh.name().unwrap_or("Mesh").to_string(),
                vertices,
                indices,
                ..Default::default()
            };
            if normals.is_none() {
                md.compute_normals();
            }
            let name = if mesh.primitives().count() > 1 {
                format!("{}.{}", md.name, i)
            } else {
                md.name.clone()
            };
            // glTF skinned vertex positions are now in armature-local bind space;
            // the mesh object itself stays at the identity so skinning does not
            // double-apply the armature transform.
            let obj_world = if skin_binding.is_some() { Mat4::IDENTITY } else { world };
            let mut obj = Object::from_matrix(name, obj_world, Arc::new(md));
            obj.material = prim.material().index();
            obj.skin = skin_binding;
            scene.objects.push(obj);
        }
    }
    for child in node.children() {
        visit(&child, world, buffers, worlds, armature_ids, armature_worlds, scene);
    }
}

#[cfg(target_arch = "wasm32")]
pub fn load_gltf(_path: &Path) -> Result<Scene> {
    anyhow::bail!("glTF load from path is not available on WebAssembly")
}
#[cfg(target_arch = "wasm32")]
pub fn load_image(_path: &Path) -> Result<TextureData> {
    anyhow::bail!("image load from path is not available on WebAssembly")
}
#[cfg(target_arch = "wasm32")]
pub fn load_environment(_path: &Path) -> Result<EnvironmentMap> {
    anyhow::bail!("environment load from path is not available on WebAssembly")
}
