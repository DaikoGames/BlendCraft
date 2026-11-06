use anyhow::Result;
use forge_core::{Material, MeshData, Object, Scene, Vertex};
use std::path::Path;

/// Wavefront OBJ import (triangulated, single index). Diffuse colors become base colors.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_obj(path: &Path) -> Result<Scene> {
    let (models, mats) = tobj::load_obj(path, &tobj::LoadOptions { triangulate: true, single_index: true, ..Default::default() })?;
    let mut scene = Scene::default();
    if let Ok(mats) = mats {
        scene.materials = mats
            .iter()
            .map(|m| {
                let d = m.diffuse.unwrap_or([0.8; 3]);
                Material { name: m.name.clone(), base_color: [d[0], d[1], d[2], 1.0], roughness: 0.8, ..Default::default() }
            })
            .collect();
    }
    for model in models {
        let me = &model.mesh;
        let n = me.positions.len() / 3;
        let has_n = me.normals.len() == me.positions.len();
        let has_t = me.texcoords.len() == n * 2;
        let vertices = (0..n)
            .map(|i| Vertex {
                position: [me.positions[i * 3], me.positions[i * 3 + 1], me.positions[i * 3 + 2]],
                normal: if has_n { [me.normals[i * 3], me.normals[i * 3 + 1], me.normals[i * 3 + 2]] } else { [0.0; 3] },
                uv: if has_t { [me.texcoords[i * 2], 1.0 - me.texcoords[i * 2 + 1]] } else { [0.0; 2] },
            })
            .collect();
        let mut md = MeshData { name: model.name.clone(), vertices, indices: me.indices.clone(), ..Default::default() };
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
