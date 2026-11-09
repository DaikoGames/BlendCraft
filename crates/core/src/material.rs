use crate::shader_graph::ShaderGraph;
use serde::{Deserialize, Serialize};

/// glTF-style metallic-roughness PBR material. Texture fields index `Scene::textures`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Material {
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub base_color_tex: Option<usize>,
    pub metallic_roughness_tex: Option<usize>,
    pub normal_tex: Option<usize>,
    pub occlusion_tex: Option<usize>,
    pub emissive_tex: Option<usize>,
    /// Optional node graph for this material (persisted in the project).
    #[serde(default)]
    pub shader_graph: Option<ShaderGraph>,
    /// Displacement scale from graph output (0 = off).
    #[serde(default)]
    pub displacement_scale: f32,
    /// Volume density placeholder from graph (renderer may ignore).
    #[serde(default)]
    pub volume_density: f32,
}

impl Material {
    /// Shift all texture indices (used when merging scenes).
    pub fn offset_textures(&mut self, off: usize) {
        for t in [
            &mut self.base_color_tex,
            &mut self.metallic_roughness_tex,
            &mut self.normal_tex,
            &mut self.occlusion_tex,
            &mut self.emissive_tex,
        ] {
            if let Some(i) = t {
                *i += off;
            }
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: "Default".into(),
            base_color: [0.72, 0.72, 0.75, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            base_color_tex: None,
            metallic_roughness_tex: None,
            normal_tex: None,
            occlusion_tex: None,
            emissive_tex: None,
            shader_graph: None,
            displacement_scale: 0.0,
            volume_density: 0.0,
        }
    }
}

/// Decoded 8-bit RGBA image. Color space is decided by how a material uses it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextureData {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}
