//! File import/export. glTF 2.0 / GLB (geometry, node transforms, PBR materials, textures)
//! and HDR environment maps.
use anyhow::{Context, Result};
use forge_core::{
    AlphaMode, EnvironmentMap, Material, MeshData, Object, Scene, TextureData, Vertex,
};
use glam::Mat4;
use std::path::Path;
use std::sync::Arc;

mod export;
mod fbx;
mod obj;
#[cfg(not(target_arch = "wasm32"))]
mod unity;
pub use export::{export_glb, export_obj};
pub use fbx::{export_fbx, load_fbx, load_fbx_bytes};
pub use obj::load_obj;
#[cfg(not(target_arch = "wasm32"))]
pub use unity::load_unitypackage;

#[cfg(not(target_arch = "wasm32"))]
pub fn load_gltf(path: &Path) -> Result<Scene> {
    let (doc, buffers, images) =
        gltf::import(path).with_context(|| format!("reading {}", path.display()))?;
    let mut scene = Scene::default();
    scene.materials = doc.materials().map(|m| convert_material(&m)).collect();
    scene.textures = Arc::new(
        doc.images()
            .zip(images.iter())
            .map(|(img, data)| convert_image(img.name().unwrap_or("Image"), data))
            .collect::<Vec<_>>(),
    );
    let gscene = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .context("file has no scene")?;
    for node in gscene.nodes() {
        visit(&node, Mat4::IDENTITY, &buffers, &mut scene);
    }
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

fn visit(node: &gltf::Node, parent: Mat4, buffers: &[gltf::buffer::Data], scene: &mut Scene) {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        for (i, prim) in mesh.primitives().enumerate() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(|b| Some(&buffers[b.index()].0[..]));
            let Some(pos) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<[f32; 3]> = pos.collect();
            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|n| n.collect());
            let uvs: Option<Vec<[f32; 2]>> =
                reader.read_tex_coords(0).map(|t| t.into_f32().collect());
            let indices: Vec<u32> = match reader.read_indices() {
                Some(idx) => idx.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            let vertices = positions
                .iter()
                .enumerate()
                .map(|(k, p)| Vertex {
                    position: *p,
                    normal: normals.as_ref().map_or([0.0; 3], |n| n[k]),
                    uv: uvs.as_ref().map_or([0.0; 2], |u| u[k]),
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
            let mut obj = Object::from_matrix(name, world, Arc::new(md));
            obj.material = prim.material().index();
            scene.objects.push(obj);
        }
    }
    for child in node.children() {
        visit(&child, world, buffers, scene);
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
