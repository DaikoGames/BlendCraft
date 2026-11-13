//! Native project format (.forge): magic + zlib(bincode(Scene)).
use crate::{
    AnimationData, Collection, ColorRampStop, Material, Object, Scene, ShaderGraph, ShaderLink,
    ShaderNode, ShaderNodeKind, TextureData,
};
use anyhow::{anyhow, bail, Result};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::sync::Arc;

const MAGIC: &[u8; 4] = b"FRG4";
const LEGACY_MAGIC: &[u8; 4] = b"FRG3";

#[derive(Serialize, Deserialize)]
struct Version3Scene {
    objects: Vec<Object>,
    materials: Vec<Version3Material>,
    textures: Arc<Vec<TextureData>>,
    collections: Vec<Collection>,
    cursor: Vec3,
    next_id: u64,
    animation: AnimationData,
}

#[derive(Serialize, Deserialize)]
struct Version3Material {
    name: String,
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    alpha_mode: crate::AlphaMode,
    alpha_cutoff: f32,
    double_sided: bool,
    normal_scale: f32,
    occlusion_strength: f32,
    base_color_tex: Option<usize>,
    metallic_roughness_tex: Option<usize>,
    normal_tex: Option<usize>,
    occlusion_tex: Option<usize>,
    emissive_tex: Option<usize>,
    shader_graph: Option<Version3ShaderGraph>,
    displacement_scale: f32,
    volume_density: f32,
}

#[derive(Serialize, Deserialize)]
struct Version3ShaderGraph {
    nodes: Vec<Version3ShaderNode>,
    links: Vec<ShaderLink>,
    next_id: u64,
}

#[derive(Serialize, Deserialize)]
struct Version3ShaderNode {
    id: u64,
    kind: ShaderNodeKind,
    pos: [f32; 2],
    image_index: Option<usize>,
    value: f32,
    color: [f32; 4],
    props: [f32; 16],
    input_defaults: Vec<[f32; 4]>,
    mode: u8,
    name_prop: String,
    ui_size: [f32; 2],
    ramp_stops: Vec<ColorRampStop>,
}

impl From<Version3Scene> for Scene {
    fn from(scene: Version3Scene) -> Self {
        Self {
            objects: scene.objects,
            materials: scene
                .materials
                .into_iter()
                .map(|material| Material {
                    name: material.name,
                    base_color: material.base_color,
                    metallic: material.metallic,
                    roughness: material.roughness,
                    emissive: material.emissive,
                    alpha_mode: material.alpha_mode,
                    alpha_cutoff: material.alpha_cutoff,
                    double_sided: material.double_sided,
                    normal_scale: material.normal_scale,
                    occlusion_strength: material.occlusion_strength,
                    base_color_tex: material.base_color_tex,
                    metallic_roughness_tex: material.metallic_roughness_tex,
                    normal_tex: material.normal_tex,
                    occlusion_tex: material.occlusion_tex,
                    emissive_tex: material.emissive_tex,
                    shader_graph: material.shader_graph.map(|graph| ShaderGraph {
                        nodes: graph
                            .nodes
                            .into_iter()
                            .map(|node| ShaderNode {
                                id: node.id,
                                kind: node.kind,
                                pos: node.pos,
                                image_index: node.image_index,
                                value: node.value,
                                color: node.color,
                                props: node.props,
                                input_defaults: node.input_defaults,
                                mode: node.mode,
                                name_prop: node.name_prop,
                                ui_size: node.ui_size,
                                ramp_stops: node.ramp_stops,
                            })
                            .collect(),
                        links: graph.links,
                        next_id: graph.next_id,
                    }),
                    displacement_scale: material.displacement_scale,
                    volume_density: material.volume_density,
                })
                .collect(),
            textures: scene.textures,
            collections: scene.collections,
            cursor: scene.cursor,
            next_id: scene.next_id,
            animation: scene.animation,
            compositor_graph: Default::default(),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct LegacyScene {
    objects: Vec<Object>,
    materials: Vec<LegacyMaterial>,
    textures: Arc<Vec<TextureData>>,
    collections: Vec<Collection>,
    cursor: Vec3,
    next_id: u64,
    animation: AnimationData,
}

#[derive(Serialize, Deserialize)]
struct LegacyMaterial {
    name: String,
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    alpha_mode: crate::AlphaMode,
    alpha_cutoff: f32,
    double_sided: bool,
    normal_scale: f32,
    occlusion_strength: f32,
    base_color_tex: Option<usize>,
    metallic_roughness_tex: Option<usize>,
    normal_tex: Option<usize>,
    occlusion_tex: Option<usize>,
    emissive_tex: Option<usize>,
    shader_graph: Option<LegacyShaderGraph>,
    displacement_scale: f32,
    volume_density: f32,
}

#[derive(Serialize, Deserialize)]
struct LegacyShaderGraph {
    nodes: Vec<LegacyShaderNode>,
    links: Vec<ShaderLink>,
    next_id: u64,
}

#[derive(Serialize, Deserialize)]
struct LegacyShaderNode {
    id: u64,
    kind: ShaderNodeKind,
    pos: [f32; 2],
    image_index: Option<usize>,
    value: f32,
    color: [f32; 4],
    props: [f32; 16],
    input_defaults: Vec<[f32; 4]>,
    mode: u8,
    name_prop: String,
}

impl From<LegacyShaderNode> for ShaderNode {
    fn from(node: LegacyShaderNode) -> Self {
        Self {
            id: node.id,
            kind: node.kind,
            pos: node.pos,
            image_index: node.image_index,
            value: node.value,
            color: node.color,
            props: node.props,
            input_defaults: node.input_defaults,
            mode: node.mode,
            name_prop: node.name_prop,
            ui_size: [0.0; 2],
            ramp_stops: Vec::new(),
        }
    }
}

impl From<LegacyScene> for Scene {
    fn from(scene: LegacyScene) -> Self {
        Self {
            objects: scene.objects,
            materials: scene
                .materials
                .into_iter()
                .map(|material| Material {
                    name: material.name,
                    base_color: material.base_color,
                    metallic: material.metallic,
                    roughness: material.roughness,
                    emissive: material.emissive,
                    alpha_mode: material.alpha_mode,
                    alpha_cutoff: material.alpha_cutoff,
                    double_sided: material.double_sided,
                    normal_scale: material.normal_scale,
                    occlusion_strength: material.occlusion_strength,
                    base_color_tex: material.base_color_tex,
                    metallic_roughness_tex: material.metallic_roughness_tex,
                    normal_tex: material.normal_tex,
                    occlusion_tex: material.occlusion_tex,
                    emissive_tex: material.emissive_tex,
                    shader_graph: material.shader_graph.map(|graph| ShaderGraph {
                        nodes: graph.nodes.into_iter().map(ShaderNode::from).collect(),
                        links: graph.links,
                        next_id: graph.next_id,
                    }),
                    displacement_scale: material.displacement_scale,
                    volume_density: material.volume_density,
                })
                .collect(),
            textures: scene.textures,
            collections: scene.collections,
            cursor: scene.cursor,
            next_id: scene.next_id,
            animation: scene.animation,
            compositor_graph: Default::default(),
        }
    }
}

pub fn to_bytes(scene: &Scene) -> Result<Vec<u8>> {
    let raw = bincode::serialize(scene)?;
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    enc.write_all(&raw)?;
    let mut out = MAGIC.to_vec();
    out.extend(enc.finish()?);
    Ok(out)
}

pub fn from_bytes(bytes: &[u8]) -> Result<Scene> {
    if bytes.len() < 4 || (&bytes[..4] != MAGIC && &bytes[..4] != LEGACY_MAGIC) {
        bail!("not a Forge3D project file");
    }
    let is_v4 = &bytes[..4] == MAGIC;
    let mut raw = Vec::new();
    flate2::read::ZlibDecoder::new(&bytes[4..]).read_to_end(&mut raw)?;
    if is_v4 {
        return Ok(bincode::deserialize(&raw)?);
    }
    let version3_error = match bincode::deserialize::<Version3Scene>(&raw) {
        Ok(scene) => return Ok(Scene::from(scene)),
        Err(error) => error,
    };
    bincode::deserialize::<LegacyScene>(&raw)
        .map(Scene::from)
        .map_err(|legacy_error| {
            anyhow!("could not decode FRG3 project: {version3_error}; {legacy_error}")
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;

    #[test]
    fn uv_coordinates_survive_project_roundtrip() {
        let mut d = Document::new();
        let obj = &mut d.scene.objects[0];
        let mesh = std::sync::Arc::make_mut(&mut obj.mesh);
        mesh.vertices[0].uv = [0.123, 0.875];
        let bytes = to_bytes(&d.scene).unwrap();
        assert_eq!(&bytes[..4], MAGIC);
        let loaded = from_bytes(&bytes).unwrap();
        let uv = loaded.objects[0].mesh.vertices[0].uv;
        assert!((uv[0] - 0.123).abs() < 1e-6);
        assert!((uv[1] - 0.875).abs() < 1e-6);
    }

    #[test]
    fn active_edit_session_uvs_are_committed_before_project_save() {
        let mut d = Document::new();
        d.select_only(0);
        d.enter_edit_mode().unwrap();
        d.edit.as_mut().unwrap().mesh.vertices[0].uv = [0.314, 0.728];
        // This is the same commit performed by the application save command.
        d.sync_edit_mesh();
        let loaded = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        let uv = loaded.objects[0].mesh.vertices[0].uv;
        assert!((uv[0] - 0.314).abs() < 1e-6);
        assert!((uv[1] - 0.728).abs() < 1e-6);
    }

    #[test]
    fn painted_texture_survives_project_roundtrip() {
        let mut d = Document::new();
        d.select_only(0);
        d.uv_unwrap_smart();
        d.enter_paint_mode().unwrap();
        assert!(d.paint_at_uv(glam::Vec2::new(0.5, 0.5), [1.0, 0.0, 0.0, 1.0], 0.025, 1.0));
        let loaded = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        let material = &loaded.materials[loaded.objects[0].material.unwrap()];
        let tex = &loaded.textures[material.base_color_tex.unwrap()];
        let center = ((tex.height / 2 * tex.width + tex.width / 2) * 4) as usize;
        assert!(tex.rgba8[center] > 240);
        assert!(tex.rgba8[center + 1] < 20);
        assert!(tex.rgba8[center + 2] < 20);
    }

    #[test]
    fn roundtrip() {
        let d = Document::new();
        let back = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        assert_eq!(back.objects.len(), 3); // Cube + Sun + Camera
        assert_eq!(
            back.objects[0].mesh.vertices.len(),
            d.scene.objects[0].mesh.vertices.len()
        );
    }

    #[test]
    fn compositor_graph_survives_project_roundtrip() {
        let mut d = Document::new();
        let effect = d
            .scene
            .compositor_graph
            .add_node(crate::CompositorNodeKind::Exposure, [260.0, 40.0]);
        d.scene
            .compositor_graph
            .nodes
            .iter_mut()
            .find(|node| node.id == effect)
            .unwrap()
            .value = 1.5;
        let loaded = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        assert_eq!(loaded.compositor_graph, d.scene.compositor_graph);
    }

    #[test]
    fn legacy_nodes_load_with_default_editor_state() {
        let legacy = LegacyScene {
            objects: Vec::new(),
            materials: vec![LegacyMaterial {
                name: "Legacy".into(),
                base_color: [1.0; 4],
                metallic: 0.0,
                roughness: 0.5,
                emissive: [0.0; 3],
                alpha_mode: crate::AlphaMode::Opaque,
                alpha_cutoff: 0.5,
                double_sided: false,
                normal_scale: 1.0,
                occlusion_strength: 1.0,
                base_color_tex: None,
                metallic_roughness_tex: None,
                normal_tex: None,
                occlusion_tex: None,
                emissive_tex: None,
                shader_graph: Some(LegacyShaderGraph {
                    nodes: vec![LegacyShaderNode {
                        id: 7,
                        kind: ShaderNodeKind::ColorRamp,
                        pos: [12.0, 34.0],
                        image_index: None,
                        value: 2.0,
                        color: [0.0, 0.0, 0.0, 1.0],
                        props: [0.0; 16],
                        input_defaults: Vec::new(),
                        mode: 0,
                        name_prop: String::new(),
                    }],
                    links: Vec::new(),
                    next_id: 8,
                }),
                displacement_scale: 0.0,
                volume_density: 0.0,
            }],
            textures: Arc::new(Vec::new()),
            collections: Vec::new(),
            cursor: Vec3::ZERO,
            next_id: 1,
            animation: AnimationData::default(),
        };
        let raw = bincode::serialize(&legacy).unwrap();
        let mut encoded = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        encoded.write_all(&raw).unwrap();
        let mut bytes = LEGACY_MAGIC.to_vec();
        bytes.extend(encoded.finish().unwrap());

        let scene = from_bytes(&bytes).unwrap();
        let node = &scene.materials[0].shader_graph.as_ref().unwrap().nodes[0];
        assert_eq!(node.id, 7);
        assert_eq!(node.pos, [12.0, 34.0]);
        assert_eq!(node.ui_size, [0.0; 2]);
        assert!(node.ramp_stops.is_empty());
    }

    #[test]
    fn frg3_project_loads_with_a_default_compositor_graph() {
        let legacy = Version3Scene {
            objects: Vec::new(),
            materials: vec![Version3Material {
                name: "Old material".into(),
                base_color: [1.0; 4],
                metallic: 0.0,
                roughness: 0.5,
                emissive: [0.0; 3],
                alpha_mode: crate::AlphaMode::Opaque,
                alpha_cutoff: 0.5,
                double_sided: false,
                normal_scale: 1.0,
                occlusion_strength: 1.0,
                base_color_tex: None,
                metallic_roughness_tex: None,
                normal_tex: None,
                occlusion_tex: None,
                emissive_tex: None,
                shader_graph: Some(Version3ShaderGraph {
                    nodes: vec![Version3ShaderNode {
                        id: 8,
                        kind: ShaderNodeKind::ColorRamp,
                        pos: [1.0, 2.0],
                        image_index: None,
                        value: 1.0,
                        color: [0.0, 0.0, 0.0, 1.0],
                        props: [0.0; 16],
                        input_defaults: Vec::new(),
                        mode: 0,
                        name_prop: String::new(),
                        ui_size: [190.0, 230.0],
                        ramp_stops: vec![ColorRampStop {
                            color: [1.0; 4],
                            position: 0.5,
                        }],
                    }],
                    links: Vec::new(),
                    next_id: 9,
                }),
                displacement_scale: 0.0,
                volume_density: 0.0,
            }],
            textures: Arc::new(Vec::new()),
            collections: Vec::new(),
            cursor: Vec3::ZERO,
            next_id: 1,
            animation: AnimationData::default(),
        };
        let raw = bincode::serialize(&legacy).unwrap();
        let mut encoded = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        encoded.write_all(&raw).unwrap();
        let mut bytes = LEGACY_MAGIC.to_vec();
        bytes.extend(encoded.finish().unwrap());

        let scene = from_bytes(&bytes).unwrap();
        assert_eq!(scene.compositor_graph.nodes.len(), 3);
        let node = &scene.materials[0].shader_graph.as_ref().unwrap().nodes[0];
        assert_eq!(node.id, 8);
        assert_eq!(node.ui_size, [190.0, 230.0]);
        assert_eq!(node.ramp_stops.len(), 1);
    }
}
