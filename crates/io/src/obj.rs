use anyhow::Result;
use forge_core::{AlphaMode, Material, MeshData, Object, Scene, TextureData, Vertex};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Strip MTL texture options (`-s 1 1 1`, `-bm 0.5`, ...) from a `map_*` value and return the
/// file name.
fn mtl_texture_name(raw: &str) -> String {
    fn arg_count(opt: &str) -> usize {
        match opt {
            "-s" | "-o" | "-t" => 3,
            "-mm" => 2,
            _ => 1, // -bm -clamp -blendu -blendv -cc -imfchan -texres -type
        }
    }
    let toks: Vec<&str> = raw.split_whitespace().collect();
    let mut i = 0;
    while i < toks.len() && toks[i].starts_with('-') {
        i += 1 + arg_count(toks[i]);
    }
    if i >= toks.len() {
        return raw.trim().to_string();
    }
    toks[i..].join(" ")
}

/// Wavefront OBJ import (triangulated, single index). Diffuse colors become base colors and
/// the textures referenced by the .mtl (diffuse, normal) are loaded from disk.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_obj(path: &Path) -> Result<Scene> {
    let (models, mats) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )?;
    let mut scene = Scene::default();
    let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let mut textures: Vec<TextureData> = Vec::new();
    let mut cache: HashMap<String, Option<usize>> = HashMap::new();
    let mut load_tex = |name: &str| -> Option<usize> {
        let name = mtl_texture_name(name);
        if name.is_empty() {
            return None;
        }
        if let Some(c) = cache.get(&name) {
            return *c;
        }
        let found = crate::find_texture_file(&dir, "", &name)
            .and_then(|b| crate::decode_image_bytes(&b, &name))
            .map(|t| {
                textures.push(t);
                textures.len() - 1
            });
        if found.is_none() {
            eprintln!("OBJ texture '{name}' was not found or could not be decoded");
        }
        cache.insert(name, found);
        found
    };
    if let Ok(mats) = mats {
        scene.materials = mats
            .iter()
            .map(|m| {
                let d = m.diffuse.unwrap_or([0.8; 3]);
                let mut mat = Material {
                    name: m.name.clone(),
                    base_color: [d[0], d[1], d[2], 1.0],
                    roughness: 0.8,
                    ..Default::default()
                };
                if let Some(t) = m.diffuse_texture.as_deref().and_then(&mut load_tex) {
                    mat.base_color_tex = Some(t);
                    // Kd multiplies the texture; many exporters leave a grey default there.
                    mat.base_color = [1.0, 1.0, 1.0, 1.0];
                }
                if let Some(t) = m.normal_texture.as_deref().and_then(&mut load_tex) {
                    mat.normal_tex = Some(t);
                }
                if let Some(a) = m.dissolve {
                    if a < 0.999 {
                        mat.base_color[3] = a.clamp(0.0, 1.0);
                        mat.alpha_mode = AlphaMode::Blend;
                    }
                }
                mat
            })
            .collect();
    }
    scene.textures = Arc::new(textures);
    for model in models {
        let me = &model.mesh;
        let n = me.positions.len() / 3;
        let has_n = me.normals.len() == me.positions.len();
        let has_t = me.texcoords.len() == n * 2;
        let vertices = (0..n)
            .map(|i| Vertex {
                position: [
                    me.positions[i * 3],
                    me.positions[i * 3 + 1],
                    me.positions[i * 3 + 2],
                ],
                normal: if has_n {
                    [
                        me.normals[i * 3],
                        me.normals[i * 3 + 1],
                        me.normals[i * 3 + 2],
                    ]
                } else {
                    [0.0; 3]
                },
                uv: if has_t {
                    [me.texcoords[i * 2], 1.0 - me.texcoords[i * 2 + 1]]
                } else {
                    [0.0; 2]
                },
                color: [1.0, 1.0, 1.0, 1.0],
            })
            .collect();
        let mut md = MeshData {
            name: model.name.clone(),
            vertices,
            indices: me.indices.clone(),
            ..Default::default()
        };
        if !has_n {
            md.compute_normals();
        }
        let mut obj = Object::new(model.name.clone(), md);
        obj.material = me.material_id.filter(|&i| i < scene.materials.len());
        scene.objects.push(obj);
    }
    Ok(scene)
}

#[cfg(target_arch = "wasm32")]
pub fn load_obj(_path: &Path) -> Result<Scene> {
    anyhow::bail!("OBJ load from path is not available on WebAssembly")
}
