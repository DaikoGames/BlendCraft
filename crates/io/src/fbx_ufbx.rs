//! FBX import via `ufbx` — meshes, materials/textures, skins, bind poses, animation.
//! Preferred path from `load_fbx`; falls back to the built-in parser on failure.

use anyhow::{bail, Result};
use forge_core::{
    AnimationClip, ArmatureData, Bone, Material, MeshData, Object, ObjectAnimation, ObjectKind,
    Scene, SkinBinding, TextureData, Vertex,
};
use glam::{Mat4, Quat, Vec3, Vec4};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub fn load_fbx_ufbx(path: &Path) -> Result<Scene> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-utf8 path"))?;
    // Convert file units → metres and axes → Y-up so size matches OBJ/GLB.
    let opts = ufbx::LoadOpts {
        target_unit_meters: 1.0,
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        ..Default::default()
    };
    let root = ufbx::load_file(path_str, opts)
        .map_err(|e| anyhow::anyhow!("ufbx load {}: {:?}", path.display(), e))?;
    convert_ufbx_scene(&root, Some(path))
}

pub fn load_fbx_ufbx_bytes(bytes: &[u8]) -> Result<Scene> {
    let opts = ufbx::LoadOpts {
        target_unit_meters: 1.0,
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        ..Default::default()
    };
    let root = ufbx::load_memory(bytes, opts)
        .map_err(|e| anyhow::anyhow!("ufbx load_memory: {:?}", e))?;
    convert_ufbx_scene(&root, None)
}

fn mat4_from_ufbx(m: ufbx::Matrix) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(m.m00 as f32, m.m10 as f32, m.m20 as f32, 0.0),
        Vec4::new(m.m01 as f32, m.m11 as f32, m.m21 as f32, 0.0),
        Vec4::new(m.m02 as f32, m.m12 as f32, m.m22 as f32, 0.0),
        Vec4::new(m.m03 as f32, m.m13 as f32, m.m23 as f32, 1.0),
    )
}

fn blob_slice(blob: &ufbx::Blob) -> &[u8] {
    // Blob: Deref<Target = [u8]>
    &**blob
}

fn decode_tex(bytes: &[u8], name: &str) -> Option<TextureData> {
    if bytes.is_empty() {
        return None;
    }
    crate::decode_image_bytes(bytes, name)
}

fn convert_ufbx_scene(fbx: &ufbx::SceneRoot, source_path: Option<&Path>) -> Result<Scene> {
    let mut scene = Scene::default();
    let dir = source_path.and_then(|p| p.parent()).unwrap_or_else(|| Path::new("."));
    let stem = source_path
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
        .unwrap_or("");

    // ------------------------------------------------------------------ textures
    // ufbx stores embedded pixels primarily on `texture_files` and sometimes on
    // Texture/Video.content. Load all sources, then wire materials by property.
    let mut tex_id_to_idx: HashMap<u32, usize> = HashMap::new();
    let mut file_index_to_idx: HashMap<u32, usize> = HashMap::new();
    let mut textures: Vec<TextureData> = Vec::new();

    let mut try_load_named = |name: &str, bytes: &[u8]| -> Option<usize> {
        if bytes.is_empty() {
            return None;
        }
        let mut td = decode_tex(bytes, name)?;
        if td.name.is_empty() {
            td.name = name.to_string();
        }
        let idx = textures.len();
        textures.push(td);
        Some(idx)
    };

    // 1) texture_files — canonical embedded / resolved file list
    for tf in &fbx.texture_files {
        let name = if !tf.filename.is_empty() {
            tf.filename.to_string()
        } else if !tf.relative_filename.is_empty() {
            tf.relative_filename.to_string()
        } else {
            format!("TextureFile_{}", tf.index)
        };
        let mut idx = try_load_named(&name, blob_slice(&tf.content));
        if idx.is_none() {
            for cand in [
                tf.relative_filename.to_string(),
                tf.filename.to_string(),
                tf.absolute_filename.to_string(),
            ] {
                if cand.is_empty() {
                    continue;
                }
                if let Some(bytes) = crate::find_texture_file(dir, stem, &cand) {
                    idx = try_load_named(&name, &bytes);
                    if idx.is_some() {
                        break;
                    }
                }
            }
        }
        if let Some(i) = idx {
            file_index_to_idx.insert(tf.index, i);
        }
    }

    // 2) Texture elements (+ linked Video)
    for tex in &fbx.textures {
        let eid = tex.element.element_id;
        if tex_id_to_idx.contains_key(&eid) {
            continue;
        }
        // Prefer already-loaded texture_files entry
        if tex.has_file {
            if let Some(&i) = file_index_to_idx.get(&tex.file_index) {
                tex_id_to_idx.insert(eid, i);
                continue;
            }
        }
        let name = if !tex.element.name.is_empty() {
            tex.element.name.to_string()
        } else if !tex.filename.is_empty() {
            tex.filename.to_string()
        } else {
            format!("Texture_{eid}")
        };
        let mut idx = try_load_named(&name, blob_slice(&tex.content));
        if idx.is_none() {
            if let Some(video) = tex.video.as_ref() {
                idx = try_load_named(&name, blob_slice(&video.content));
                if idx.is_none() {
                    for cand in [
                        video.relative_filename.to_string(),
                        video.filename.to_string(),
                        video.absolute_filename.to_string(),
                    ] {
                        if cand.is_empty() {
                            continue;
                        }
                        if let Some(bytes) = crate::find_texture_file(dir, stem, &cand) {
                            idx = try_load_named(&name, &bytes);
                            if idx.is_some() {
                                break;
                            }
                        }
                    }
                }
            }
        }
        if idx.is_none() {
            for cand in [
                tex.relative_filename.to_string(),
                tex.filename.to_string(),
                tex.absolute_filename.to_string(),
            ] {
                if cand.is_empty() {
                    continue;
                }
                if let Some(bytes) = crate::find_texture_file(dir, stem, &cand) {
                    idx = try_load_named(&name, &bytes);
                    if idx.is_some() {
                        break;
                    }
                }
            }
        }
        if let Some(i) = idx {
            tex_id_to_idx.insert(eid, i);
        } else {
            eprintln!("FBX: texture '{name}' not loaded (file='{}')", tex.filename);
        }
    }

    // 3) Orphan videos
    for video in &fbx.videos {
        let eid = video.element.element_id;
        if tex_id_to_idx.contains_key(&eid) {
            continue;
        }
        let name = if !video.element.name.is_empty() {
            video.element.name.to_string()
        } else {
            video.filename.to_string()
        };
        if let Some(i) = try_load_named(&name, blob_slice(&video.content)) {
            tex_id_to_idx.insert(eid, i);
        }
    }

    scene.textures = Arc::new(textures);
    eprintln!(
        "FBX: loaded {} texture(s) into scene",
        scene.textures.len()
    );

    // ---------------------------------------------------------------- materials
    let mut mat_id_to_idx: HashMap<u32, usize> = HashMap::new();
    for mat in &fbx.materials {
        let mut m = Material::default();
        m.name = mat.element.name.to_string();

        let base = &mat.pbr.base_color;
        let diffuse = &mat.fbx.diffuse_color;
        if base.has_value {
            let c = base.value_vec4;
            m.base_color = [c.x as f32, c.y as f32, c.z as f32, c.w as f32];
        } else if diffuse.has_value {
            let c = diffuse.value_vec4;
            m.base_color = [c.x as f32, c.y as f32, c.z as f32, c.w as f32];
        }
        if mat.pbr.metalness.has_value {
            m.metallic = mat.pbr.metalness.value_vec4.x as f32;
        }
        if mat.pbr.roughness.has_value {
            m.roughness = mat.pbr.roughness.value_vec4.x as f32;
        }
        if mat.fbx.emission_color.has_value {
            let c = mat.fbx.emission_color.value_vec4;
            m.emissive = [c.x as f32, c.y as f32, c.z as f32];
        }

        // Resolve albedo texture from PBR / FBX maps, then from material.textures list.
        let mut albedo_tex: Option<usize> = None;
        let map_tex = |map: &ufbx::MaterialMap| -> Option<usize> {
            let tex = map.texture.as_ref()?;
            tex_id_to_idx
                .get(&tex.element.element_id)
                .copied()
                .or_else(|| {
                    tex.video
                        .as_ref()
                        .and_then(|v| tex_id_to_idx.get(&v.element.element_id).copied())
                })
                .or_else(|| {
                    if tex.has_file {
                        file_index_to_idx.get(&tex.file_index).copied()
                    } else {
                        None
                    }
                })
        };
        albedo_tex = map_tex(base)
            .or_else(|| map_tex(diffuse))
            .or_else(|| map_tex(&mat.fbx.diffuse_factor));

        if albedo_tex.is_none() {
            for mt in &mat.textures {
                let prop = mt.material_prop.to_ascii_lowercase();
                let shader = mt.shader_prop.to_ascii_lowercase();
                let is_albedo = prop.contains("diffuse")
                    || prop.contains("basecolor")
                    || prop.contains("base_color")
                    || prop == "albedo"
                    || shader.contains("diffuse")
                    || shader.contains("basecolor")
                    || shader.contains("base_color");
                if !is_albedo {
                    continue;
                }
                let tex = &mt.texture;
                albedo_tex = tex_id_to_idx
                    .get(&tex.element.element_id)
                    .copied()
                    .or_else(|| {
                        if tex.has_file {
                            file_index_to_idx.get(&tex.file_index).copied()
                        } else {
                            None
                        }
                    });
                if albedo_tex.is_some() {
                    break;
                }
            }
        }
        // Last resort: first texture on the material
        if albedo_tex.is_none() {
            for mt in &mat.textures {
                let tex = &mt.texture;
                albedo_tex = tex_id_to_idx
                    .get(&tex.element.element_id)
                    .copied()
                    .or_else(|| {
                        if tex.has_file {
                            file_index_to_idx.get(&tex.file_index).copied()
                        } else {
                            None
                        }
                    });
                if albedo_tex.is_some() {
                    break;
                }
            }
        }
        m.base_color_tex = albedo_tex;
        // Textured materials with a zero/near-black factor would look black in the
        // viewport (shader multiplies factor * texture). Prefer a visible factor.
        if m.base_color_tex.is_some() {
            let c = m.base_color;
            if c[0] + c[1] + c[2] < 0.05 {
                m.base_color = [1.0, 1.0, 1.0, 1.0];
            }
        }

        if let Some(ti) = map_tex(&mat.fbx.normal_map).or_else(|| map_tex(&mat.fbx.bump)) {
            m.normal_tex = Some(ti);
        }

        mat_id_to_idx.insert(mat.element.element_id, scene.materials.len());
        scene.materials.push(m);
    }
    if scene.materials.is_empty() {
        scene.materials.push(Material::default());
    }
    eprintln!(
        "FBX: {} material(s), {} with base_color_tex",
        scene.materials.len(),
        scene
            .materials
            .iter()
            .filter(|m| m.base_color_tex.is_some())
            .count()
    );

    // -------------------------------------------------------------------- bones
    let mut node_to_bone: HashMap<u32, u32> = HashMap::new();
    let mut bone_nodes: Vec<&ufbx::Node> = Vec::new();

    for skin in &fbx.skin_deformers {
        for cluster in &skin.clusters {
            let Some(bone_node) = cluster.bone_node.as_ref() else {
                continue;
            };
            let eid = bone_node.element.element_id;
            if node_to_bone.contains_key(&eid) {
                continue;
            }
            let bi = bone_nodes.len() as u32;
            node_to_bone.insert(eid, bi);
            bone_nodes.push(bone_node);
        }
    }
    for node in &fbx.nodes {
        if node.bone.is_some() {
            let eid = node.element.element_id;
            if !node_to_bone.contains_key(&eid) {
                let bi = bone_nodes.len() as u32;
                node_to_bone.insert(eid, bi);
                bone_nodes.push(node);
            }
        }
    }

    let mut armature_id: Option<u64> = None;
    if !bone_nodes.is_empty() {
        let mut bind_global: Vec<Mat4> = vec![Mat4::IDENTITY; bone_nodes.len()];
        for skin in &fbx.skin_deformers {
            for cluster in &skin.clusters {
                let Some(bone_node) = cluster.bone_node.as_ref() else {
                    continue;
                };
                let eid = bone_node.element.element_id;
                let Some(&bi) = node_to_bone.get(&eid) else {
                    continue;
                };
                let g2b = mat4_from_ufbx(cluster.geometry_to_bone);
                if g2b.determinant().abs() > 1e-12 {
                    bind_global[bi as usize] = g2b.inverse();
                }
            }
        }
        for (bi, node) in bone_nodes.iter().enumerate() {
            if bind_global[bi] == Mat4::IDENTITY {
                bind_global[bi] = mat4_from_ufbx(node.node_to_world);
            }
        }

        let mut arm = ArmatureData::default();
        arm.bones.clear();
        for (bi, node) in bone_nodes.iter().enumerate() {
            let parent_bi = node
                .parent
                .as_ref()
                .and_then(|p| node_to_bone.get(&p.element.element_id).copied());
            let local = if let Some(p) = parent_bi {
                bind_global[p as usize].inverse() * bind_global[bi]
            } else {
                bind_global[bi]
            };
            let (scale, rot, trans) = local.to_scale_rotation_translation();
            let head = bind_global[bi].transform_point3(Vec3::ZERO);
            let (_, orient, _) = bind_global[bi].to_scale_rotation_translation();
            let orient = orient.normalize();
            let local_y = (orient * Vec3::Y).normalize_or_zero();
            let mut tail = if local_y.length_squared() > 1e-8 {
                head + local_y * 0.1
            } else {
                head + Vec3::Y * 0.1
            };
            for (ci, child) in bone_nodes.iter().enumerate() {
                if child.parent.as_ref().map(|p| p.element.element_id)
                    == Some(node.element.element_id)
                {
                    let ch = bind_global[ci].transform_point3(Vec3::ZERO);
                    if (ch - head).length_squared() > 1e-10 {
                        tail = ch;
                        break;
                    }
                }
            }
            let mut bone = Bone::new(node.element.name.to_string(), head, tail);
            bone.parent = parent_bi;
            bone.rest_orientation = Some(orient.to_array());
            bone.rest_local_loc = Some(trans.to_array());
            bone.rest_local_rot = Some(rot.normalize().to_array());
            bone.rest_local_scale = Some(scale.to_array());
            bone.pose_loc = trans.to_array();
            bone.pose_rot = rot.normalize().to_array();
            bone.pose_scale = scale.to_array();
            arm.bones.push(bone);
        }
        if !arm.bones.is_empty() {
            arm.active = Some(0);
            arm.selected.insert(0);
            let mut arm_obj = Object::new("Armature", MeshData::default());
            arm_obj.id = scene.alloc_id();
            arm_obj.kind = ObjectKind::Armature(arm);
            armature_id = Some(arm_obj.id);
            scene.objects.push(arm_obj);
        }
    }

    // -------------------------------------------------------------------- meshes
    for node in &fbx.nodes {
        let Some(mesh) = node.mesh.as_ref() else {
            continue;
        };
        if mesh.num_faces == 0 {
            continue;
        }

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut corner_cp: Vec<u32> = Vec::new();

        for face_ix in 0..mesh.num_faces {
            let face = mesh.faces[face_ix as usize];
            let n = face.num_indices as usize;
            if n < 3 {
                continue;
            }
            let mut corners: Vec<u32> = Vec::with_capacity(n);
            for k in 0..n {
                corners.push(mesh.vertex_indices[(face.index_begin as usize) + k]);
            }
            for t in 1..corners.len().saturating_sub(1) {
                for &ci in &[0usize, t, t + 1] {
                    let corner_index = (face.index_begin as usize) + ci;
                    let cp = corners[ci];
                    let p = mesh.vertex_position[corner_index];
                    positions.push([p.x as f32, p.y as f32, p.z as f32]);
                    if !mesh.vertex_normal.values.is_empty() {
                        let nv = mesh.vertex_normal[corner_index];
                        normals.push([nv.x as f32, nv.y as f32, nv.z as f32]);
                    } else {
                        normals.push([0.0, 1.0, 0.0]);
                    }
                    // FBX UVs: V=0 at bottom. Image buffers + our sampler use top-left
                    // (same as the built-in FBX path which stores 1-V).
                    if !mesh.vertex_uv.values.is_empty() {
                        let u = mesh.vertex_uv[corner_index];
                        uvs.push([u.x as f32, 1.0 - u.y as f32]);
                    } else if !mesh.uv_sets.is_empty()
                        && !mesh.uv_sets[0].vertex_uv.values.is_empty()
                    {
                        let u = mesh.uv_sets[0].vertex_uv[corner_index];
                        uvs.push([u.x as f32, 1.0 - u.y as f32]);
                    } else {
                        uvs.push([0.0, 0.0]);
                    }
                    corner_cp.push(cp);
                    indices.push(positions.len() as u32 - 1);
                }
            }
        }
        if positions.is_empty() {
            continue;
        }

        let mut skin_binding: Option<SkinBinding> = None;
        if let Some(arm_id) = armature_id {
            if let Some(skin_ref) = mesh.skin_deformers.first() {
                let skin = skin_ref.as_ref();
                let arm_bone_count = scene
                    .objects
                    .iter()
                    .find(|o| o.id == arm_id)
                    .and_then(|o| match &o.kind {
                        ObjectKind::Armature(a) => Some(a.bones.len()),
                        _ => None,
                    })
                    .unwrap_or(0);

                let mut cluster_to_bone: Vec<u16> = Vec::with_capacity(skin.clusters.len());
                let mut inverse_binds = vec![Mat4::IDENTITY.to_cols_array(); arm_bone_count];
                for cluster in &skin.clusters {
                    let bone_idx = cluster
                        .bone_node
                        .as_ref()
                        .and_then(|n| node_to_bone.get(&n.element.element_id).copied())
                        .unwrap_or(0);
                    cluster_to_bone.push(bone_idx as u16);
                    let g2b = mat4_from_ufbx(cluster.geometry_to_bone);
                    if (bone_idx as usize) < inverse_binds.len() {
                        inverse_binds[bone_idx as usize] = g2b.to_cols_array();
                    }
                }

                let mut joints = vec![[0u16; 4]; positions.len()];
                let mut weights = vec![[0f32; 4]; positions.len()];
                for (out_i, &cp) in corner_cp.iter().enumerate() {
                    let cp = cp as usize;
                    if cp >= skin.vertices.len() {
                        continue;
                    }
                    let sv = skin.vertices[cp];
                    let mut pairs: Vec<(u16, f32)> = Vec::new();
                    for w in 0..sv.num_weights {
                        let sw = skin.weights[sv.weight_begin as usize + w as usize];
                        let bi = *cluster_to_bone.get(sw.cluster_index as usize).unwrap_or(&0);
                        pairs.push((bi, sw.weight as f32));
                    }
                    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    pairs.truncate(4);
                    let sum: f32 = pairs.iter().map(|p| p.1).sum::<f32>().max(1e-8);
                    for (k, (bi, w)) in pairs.iter().enumerate() {
                        joints[out_i][k] = *bi;
                        weights[out_i][k] = *w / sum;
                    }
                }

                skin_binding = Some(SkinBinding {
                    armature_id: arm_id,
                    joints,
                    weights,
                    rest_positions: positions.clone(),
                    rest_normals: normals.clone(),
                    inverse_binds,
                });
            }
        }

        let vertices: Vec<Vertex> = positions
            .iter()
            .enumerate()
            .map(|(i, p)| Vertex {
                position: *p,
                normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
                uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                color: [1.0, 1.0, 1.0, 1.0],
            })
            .collect();

        let index_count = indices.len() as u32;
        let mut md = MeshData {
            name: node.element.name.to_string(),
            vertices,
            indices,
            ..Default::default()
        };
        md.compute_normals();

        let world = mat4_from_ufbx(node.node_to_world);
        let obj_world = if skin_binding.is_some() {
            Mat4::IDENTITY
        } else {
            world
        };
        // Material: node first, then mesh materials, then first scene material.
        let mat_idx = node
            .materials
            .iter()
            .find_map(|m| mat_id_to_idx.get(&m.element.element_id).copied())
            .or_else(|| {
                mesh.materials
                    .iter()
                    .find_map(|m| mat_id_to_idx.get(&m.element.element_id).copied())
            })
            .or_else(|| {
                if scene.materials.is_empty() {
                    None
                } else {
                    Some(0)
                }
            });

        // One full-mesh group so viewport/Cycles never fall back to "default"
        // grey when the object material slot is ignored by some draw paths.
        if let Some(mi) = mat_idx {
            if index_count > 0 {
                md.groups.push(forge_core::MeshGroup {
                    first_index: 0,
                    index_count,
                    material: Some(mi),
                });
            }
        }

        let mut obj = Object::from_matrix(node.element.name.to_string(), obj_world, Arc::new(md));
        obj.material = mat_idx;
        obj.skin = skin_binding;
        if let Some(arm_id) = armature_id {
            if obj.skin.is_some() {
                obj.parent = Some(arm_id);
            }
        }
        scene.objects.push(obj);
    }

    if scene.objects.is_empty() {
        bail!("FBX contained no usable mesh or armature objects");
    }

    // --------------------------------------------------------------- animation
    let fps = scene.animation.fps.max(1) as f64;
    for (stack_i, stack) in fbx.anim_stacks.iter().enumerate() {
        let time_begin = stack.time_begin;
        let time_end = stack.time_end;
        if !(time_end > time_begin) {
            continue;
        }
        let n_frames = (((time_end - time_begin) * fps).ceil() as i32).max(1);
        let frame_start = 1i32;
        let frame_end = frame_start + n_frames;
        let clip_name = if stack.element.name.is_empty() {
            format!("AnimStack {}", stack_i + 1)
        } else {
            stack.element.name.to_string()
        };
        scene.animation.clips.push(AnimationClip {
            name: clip_name,
            frame_start,
            frame_end,
        });

        let mut clip_tracks: HashMap<u64, HashMap<u32, ObjectAnimation>> = HashMap::new();
        if let Some(arm_id) = armature_id {
            let anim = &stack.anim;
            for fi in 0..=n_frames {
                let time = time_begin + (fi as f64) / fps;
                let frame = frame_start + fi;
                for (bi, node) in bone_nodes.iter().enumerate() {
                    let xform = node.evaluate_transform(anim, time);
                    let loc = Vec3::new(
                        xform.translation.x as f32,
                        xform.translation.y as f32,
                        xform.translation.z as f32,
                    );
                    let rot = Quat::from_xyzw(
                        xform.rotation.x as f32,
                        xform.rotation.y as f32,
                        xform.rotation.z as f32,
                        xform.rotation.w as f32,
                    )
                    .normalize();
                    let scl = Vec3::new(
                        xform.scale.x as f32,
                        xform.scale.y as f32,
                        xform.scale.z as f32,
                    );
                    let arm_map = clip_tracks.entry(arm_id).or_default();
                    let track = arm_map.entry(bi as u32).or_default();
                    track.insert_or_merge(frame, Some(loc), Some(rot), Some(scl));
                }
            }
        }
        scene.animation.bone_tracks.push(clip_tracks);
    }
    if !scene.animation.clips.is_empty() {
        scene.animation.select_clip(Some(0));
    }

    // ---------------------------------------------------------- world size = GLB (metres)
    // Do NOT rewrite vertices / IBM / bone locals (that breaks skinning).
    // Scale only object transforms so the whole FBX matches glTF metres in the viewport.
    let unit = fbx.settings.unit_meters as f32;
    let original = fbx.settings.original_unit_meters as f32;
    let mut max_abs = 0.0f32;
    for o in &scene.objects {
        for v in &o.mesh.vertices {
            max_abs = max_abs
                .max(v.position[0].abs())
                .max(v.position[1].abs())
                .max(v.position[2].abs());
        }
        if let ObjectKind::Armature(arm) = &o.kind {
            for b in &arm.bones {
                max_abs = max_abs
                    .max(b.head[0].abs())
                    .max(b.head[1].abs())
                    .max(b.head[2].abs())
                    .max(b.tail[0].abs())
                    .max(b.tail[1].abs())
                    .max(b.tail[2].abs());
            }
        }
    }
    let s = if unit.is_finite() && unit > 1e-9 && (unit - 1.0).abs() > 0.05 {
        unit
    } else if original.is_finite() && original > 1e-9 && original < 0.5 && max_abs > 5.0 {
        original
    } else if max_abs > 10.0 {
        0.01
    } else {
        1.0
    };
    if (s - 1.0).abs() > 1e-6 {
        eprintln!(
            "FBX→GLB size: object scale ×{s} (unit_meters={unit}, original={original}, extent={max_abs:.3})"
        );
        // Prefer scaling the armature root; skinned meshes are parented to it.
        // Unparented meshes get the same scale so static FBX meshes match too.
        let arm_ids: std::collections::HashSet<u64> = scene
            .objects
            .iter()
            .filter(|o| matches!(o.kind, ObjectKind::Armature(_)))
            .map(|o| o.id)
            .collect();
        for o in &mut scene.objects {
            let under_arm = o.parent.map(|p| arm_ids.contains(&p)).unwrap_or(false);
            if matches!(o.kind, ObjectKind::Armature(_)) {
                o.scale *= s;
                o.translation *= s;
            } else if o.skin.is_some() && under_arm {
                // inherits armature scale — leave local scale at 1
            } else if o.skin.is_some() || o.kind.is_mesh() {
                o.scale *= s;
                o.translation *= s;
            }
        }
    }

    scene.fix_ids();
    Ok(scene)
}
