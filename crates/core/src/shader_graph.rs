//! Serializable material shader graph and CPU evaluation into PBR slots / textures.
use crate::material::{Material, TextureData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShaderNodeKind {
    Output,
    Principled,
    ImageTexture,
    Rgb,
    TexCoord,
    Value,
    Mapping,
    Noise,
    Voronoi,
    ColorRamp,
    Mix,
    Bump,
    NormalMap,
    Emission,
    Glossy,
    Diffuse,
}

impl ShaderNodeKind {
    pub fn title(self) -> &'static str {
        match self {
            Self::Output => "Material Output",
            Self::Principled => "Principled BSDF",
            Self::ImageTexture => "Image Texture",
            Self::Rgb => "RGB",
            Self::TexCoord => "Texture Coordinate",
            Self::Value => "Value",
            Self::Mapping => "Mapping",
            Self::Noise => "Noise Texture",
            Self::Voronoi => "Voronoi Texture",
            Self::ColorRamp => "Color Ramp",
            Self::Mix => "Mix",
            Self::Bump => "Bump",
            Self::NormalMap => "Normal Map",
            Self::Emission => "Emission",
            Self::Glossy => "Glossy BSDF",
            Self::Diffuse => "Diffuse BSDF",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShaderNode {
    pub id: u64,
    pub kind: ShaderNodeKind,
    pub pos: [f32; 2],
    pub image_index: Option<usize>,
    pub value: f32,
    /// RGBA 0–1 for RGB / Emission color nodes.
    pub color: [f32; 4],
}

impl Default for ShaderNode {
    fn default() -> Self {
        Self {
            id: 0,
            kind: ShaderNodeKind::Value,
            pos: [0.0, 0.0],
            image_index: None,
            value: 0.5,
            color: [0.8, 0.8, 0.8, 1.0],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShaderLink {
    pub from: u64,
    pub to: u64,
    /// Optional socket names; empty = default primary sockets.
    #[serde(default)]
    pub from_socket: u8,
    #[serde(default)]
    pub to_socket: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShaderGraph {
    pub nodes: Vec<ShaderNode>,
    pub links: Vec<ShaderLink>,
    pub next_id: u64,
}

impl Default for ShaderGraph {
    fn default() -> Self {
        Self {
            nodes: vec![
                ShaderNode {
                    id: 1,
                    kind: ShaderNodeKind::Principled,
                    pos: [330.0, 70.0],
                    ..Default::default()
                },
                ShaderNode {
                    id: 2,
                    kind: ShaderNodeKind::Output,
                    pos: [700.0, 105.0],
                    ..Default::default()
                },
            ],
            links: vec![ShaderLink {
                from: 1,
                to: 2,
                from_socket: 0,
                to_socket: 0,
            }],
            next_id: 3,
        }
    }
}

/// Result of evaluating a graph into material slots (and optional new textures).
#[derive(Clone, Debug, Default)]
pub struct GraphEvalResult {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub base_color_tex: Option<usize>,
    pub metallic_roughness_tex: Option<usize>,
    pub normal_tex: Option<usize>,
    pub occlusion_tex: Option<usize>,
    pub emissive_tex: Option<usize>,
    /// Newly generated procedural textures to append to the scene.
    pub new_textures: Vec<TextureData>,
    pub displacement_scale: f32,
    pub volume_density: f32,
}

fn hash2(x: f32, y: f32) -> f32 {
    let mut n = x * 127.1 + y * 311.7;
    n = (n.sin() * 43758.5453).fract();
    n
}

fn noise2(x: f32, y: f32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let fx = x.fract();
    let fy = y.fract();
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let a = hash2(ix, iy);
    let b = hash2(ix + 1.0, iy);
    let c = hash2(ix, iy + 1.0);
    let d = hash2(ix + 1.0, iy + 1.0);
    a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy
}

fn fbm(x: f32, y: f32, octaves: i32) -> f32 {
    let mut v = 0.0;
    let mut a = 0.5;
    let mut f = 1.0;
    let mut x = x;
    let mut y = y;
    for _ in 0..octaves {
        v += a * noise2(x * f, y * f);
        a *= 0.5;
        f *= 2.0;
        x += 13.1;
        y += 7.7;
    }
    v.clamp(0.0, 1.0)
}

fn voronoi(x: f32, y: f32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let fx = x.fract();
    let fy = y.fract();
    let mut md = 1.0f32;
    for j in -1..=1 {
        for i in -1..=1 {
            let ox = hash2(ix + i as f32, iy + j as f32);
            let oy = hash2(ix + i as f32 + 19.2, iy + j as f32 + 7.4);
            let dx = (i as f32) + ox - fx;
            let dy = (j as f32) + oy - fy;
            md = md.min(dx * dx + dy * dy);
        }
    }
    md.sqrt().clamp(0.0, 1.0)
}

fn sample_tex(tex: &TextureData, u: f32, v: f32) -> [f32; 4] {
    if tex.width == 0 || tex.height == 0 || tex.rgba8.len() < 4 {
        return [1.0, 1.0, 1.0, 1.0];
    }
    let u = u.fract().rem_euclid(1.0);
    let v = v.fract().rem_euclid(1.0);
    let x = ((u * tex.width as f32) as u32).min(tex.width - 1);
    let y = ((v * tex.height as f32) as u32).min(tex.height - 1);
    let i = ((y * tex.width + x) * 4) as usize;
    if i + 3 >= tex.rgba8.len() {
        return [1.0, 1.0, 1.0, 1.0];
    }
    [
        tex.rgba8[i] as f32 / 255.0,
        tex.rgba8[i + 1] as f32 / 255.0,
        tex.rgba8[i + 2] as f32 / 255.0,
        tex.rgba8[i + 3] as f32 / 255.0,
    ]
}

fn color_ramp(t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    // Simple black→white ramp (extensible later).
    [t, t, t, 1.0]
}

fn mix4(a: [f32; 4], b: [f32; 4], f: f32) -> [f32; 4] {
    let f = f.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
        a[3] + (b[3] - a[3]) * f,
    ]
}

#[derive(Clone, Copy)]
enum SocketVal {
    Scalar(f32),
    Color([f32; 4]),
    Vector([f32; 3]),
}

impl SocketVal {
    fn as_scalar(self) -> f32 {
        match self {
            Self::Scalar(s) => s,
            Self::Color(c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2],
            Self::Vector(v) => v[0],
        }
    }
    fn as_color(self) -> [f32; 4] {
        match self {
            Self::Scalar(s) => [s, s, s, 1.0],
            Self::Color(c) => c,
            Self::Vector(v) => [v[0], v[1], v[2], 1.0],
        }
    }
    fn as_vector(self) -> [f32; 3] {
        match self {
            Self::Scalar(s) => [s, s, s],
            Self::Color(c) => [c[0], c[1], c[2]],
            Self::Vector(v) => v,
        }
    }
}

fn find_input(
    graph: &ShaderGraph,
    node_id: u64,
    socket: u8,
    cache: &HashMap<(u64, u8), SocketVal>,
) -> Option<SocketVal> {
    for link in &graph.links {
        if link.to == node_id && link.to_socket == socket {
            return cache.get(&(link.from, link.from_socket)).copied();
        }
    }
    None
}

fn topo_order(graph: &ShaderGraph) -> Vec<u64> {
    let mut indeg: HashMap<u64, usize> = graph.nodes.iter().map(|n| (n.id, 0)).collect();
    let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
    for l in &graph.links {
        *indeg.entry(l.to).or_insert(0) += 1;
        adj.entry(l.from).or_default().push(l.to);
    }
    let mut q: Vec<u64> = indeg
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(id, _)| *id)
        .collect();
    let mut out = Vec::new();
    while let Some(id) = q.pop() {
        out.push(id);
        if let Some(nexts) = adj.get(&id) {
            for &n in nexts {
                if let Some(d) = indeg.get_mut(&n) {
                    *d = d.saturating_sub(1);
                    if *d == 0 {
                        q.push(n);
                    }
                }
            }
        }
    }
    // Append any remaining (cycles) so evaluation still runs something.
    for n in &graph.nodes {
        if !out.contains(&n.id) {
            out.push(n.id);
        }
    }
    out
}

/// Evaluate the graph against existing scene textures. Procedural results that need
/// a texture are generated at `tex_size`² and returned in `new_textures`.
pub fn evaluate_graph(
    graph: &ShaderGraph,
    material: &Material,
    textures: &[TextureData],
    tex_size: u32,
) -> GraphEvalResult {
    let mut result = GraphEvalResult {
        base_color: material.base_color,
        metallic: material.metallic,
        roughness: material.roughness,
        emissive: material.emissive,
        normal_scale: material.normal_scale,
        occlusion_strength: material.occlusion_strength,
        base_color_tex: material.base_color_tex,
        metallic_roughness_tex: material.metallic_roughness_tex,
        normal_tex: material.normal_tex,
        occlusion_tex: material.occlusion_tex,
        emissive_tex: material.emissive_tex,
        new_textures: Vec::new(),
        displacement_scale: 0.0,
        volume_density: 0.0,
    };

    let order = topo_order(graph);
    let node_map: HashMap<u64, &ShaderNode> = graph.nodes.iter().map(|n| (n.id, n)).collect();
    let mut cache: HashMap<(u64, u8), SocketVal> = HashMap::new();
    // For procedural texture nodes we may bake a full image once.
    let mut proc_tex_index: HashMap<u64, usize> = HashMap::new();

    for id in order {
        let Some(node) = node_map.get(&id) else { continue };
        match node.kind {
            ShaderNodeKind::Value => {
                cache.insert((id, 0), SocketVal::Scalar(node.value));
            }
            ShaderNodeKind::Rgb => {
                cache.insert((id, 0), SocketVal::Color(node.color));
            }
            ShaderNodeKind::TexCoord => {
                // UV is supplied per-sample when baking; as a constant we expose (0.5,0.5,0).
                cache.insert((id, 0), SocketVal::Vector([0.5, 0.5, 0.0]));
            }
            ShaderNodeKind::Mapping => {
                let inp = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Vector([0.5, 0.5, 0.0]))
                    .as_vector();
                let s = if node.value.abs() < 1e-6 { 1.0 } else { node.value };
                cache.insert((id, 0), SocketVal::Vector([inp[0] * s, inp[1] * s, inp[2] * s]));
            }
            ShaderNodeKind::ImageTexture => {
                if let Some(ti) = node.image_index.filter(|i| *i < textures.len()) {
                    // Expose average color as scalar/color fallback; real UV sampling happens in bake.
                    let tex = &textures[ti];
                    let c = sample_tex(tex, 0.5, 0.5);
                    cache.insert((id, 0), SocketVal::Color(c));
                    // Remember index for material slot wiring.
                    proc_tex_index.insert(id, ti);
                } else {
                    cache.insert((id, 0), SocketVal::Color([1.0, 1.0, 1.0, 1.0]));
                }
            }
            ShaderNodeKind::Noise | ShaderNodeKind::Voronoi => {
                // Bake a procedural texture once.
                let size = tex_size.max(64).min(1024);
                let scale = if node.value.abs() < 1e-4 { 5.0 } else { node.value };
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32;
                        let v = y as f32 / size as f32;
                        let n = if node.kind == ShaderNodeKind::Noise {
                            fbm(u * scale, v * scale, 5)
                        } else {
                            voronoi(u * scale, v * scale)
                        };
                        let i = ((y * size + x) * 4) as usize;
                        let b = (n * 255.0) as u8;
                        rgba[i] = b;
                        rgba[i + 1] = b;
                        rgba[i + 2] = b;
                        rgba[i + 3] = 255;
                    }
                }
                let name = if node.kind == ShaderNodeKind::Noise {
                    format!("Noise_{id}")
                } else {
                    format!("Voronoi_{id}")
                };
                let tex = TextureData {
                    name,
                    width: size,
                    height: size,
                    rgba8: rgba,
                };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Color(mid));
            }
            ShaderNodeKind::ColorRamp => {
                let fac = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Scalar(0.5))
                    .as_scalar();
                cache.insert((id, 0), SocketVal::Color(color_ramp(fac)));
            }
            ShaderNodeKind::Mix => {
                let fac = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Scalar(0.5))
                    .as_scalar();
                let a = find_input(graph, id, 1, &cache)
                    .unwrap_or(SocketVal::Color([0.0, 0.0, 0.0, 1.0]))
                    .as_color();
                let b = find_input(graph, id, 2, &cache)
                    .unwrap_or(SocketVal::Color([1.0, 1.0, 1.0, 1.0]))
                    .as_color();
                cache.insert((id, 0), SocketVal::Color(mix4(a, b, fac)));
            }
            ShaderNodeKind::Bump => {
                let h = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Scalar(0.5))
                    .as_scalar();
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 1.0]));
                result.normal_scale = (h * node.value).clamp(0.0, 2.0);
            }
            ShaderNodeKind::NormalMap => {
                if let Some(c) = find_input(graph, id, 0, &cache) {
                    let _ = c;
                }
                result.normal_scale = node.value.clamp(0.0, 2.0);
                // If upstream image, use as normal tex.
                for link in &graph.links {
                    if link.to == id {
                        if let Some(&ti) = proc_tex_index.get(&link.from) {
                            result.normal_tex = Some(ti);
                        }
                    }
                }
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 1.0]));
            }
            ShaderNodeKind::Emission => {
                let col = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Color(node.color))
                    .as_color();
                let s = node.value.max(0.0);
                cache.insert((id, 0), SocketVal::Color([col[0] * s, col[1] * s, col[2] * s, 1.0]));
                result.emissive = [col[0] * s, col[1] * s, col[2] * s];
            }
            ShaderNodeKind::Diffuse => {
                let col = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Color(node.color))
                    .as_color();
                cache.insert((id, 0), SocketVal::Color(col));
                result.base_color = col;
                result.metallic = 0.0;
                result.roughness = 0.9;
            }
            ShaderNodeKind::Glossy => {
                let col = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Color(node.color))
                    .as_color();
                cache.insert((id, 0), SocketVal::Color(col));
                result.base_color = col;
                result.metallic = 1.0;
                result.roughness = node.value.clamp(0.045, 1.0);
            }
            ShaderNodeKind::Principled => {
                if let Some(c) = find_input(graph, id, 0, &cache) {
                    result.base_color = c.as_color();
                }
                // Wire image textures into base_color_tex when linked to base color.
                for link in &graph.links {
                    if link.to == id {
                        if let Some(&ti) = proc_tex_index.get(&link.from) {
                            if let Some(src) = node_map.get(&link.from) {
                                match src.kind {
                                    ShaderNodeKind::ImageTexture
                                    | ShaderNodeKind::Noise
                                    | ShaderNodeKind::Voronoi
                                    | ShaderNodeKind::Mix
                                    | ShaderNodeKind::ColorRamp => {
                                        result.base_color_tex = Some(ti);
                                    }
                                    ShaderNodeKind::NormalMap => {
                                        result.normal_tex = Some(ti);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                if let Some(m) = find_input(graph, id, 1, &cache) {
                    result.metallic = m.as_scalar().clamp(0.0, 1.0);
                }
                if let Some(r) = find_input(graph, id, 2, &cache) {
                    result.roughness = r.as_scalar().clamp(0.045, 1.0);
                }
                if let Some(e) = find_input(graph, id, 3, &cache) {
                    let c = e.as_color();
                    result.emissive = [c[0], c[1], c[2]];
                }
                cache.insert((id, 0), SocketVal::Color(result.base_color));
            }
            ShaderNodeKind::Output => {
                // Surface (socket 0)
                if let Some(s) = find_input(graph, id, 0, &cache) {
                    let c = s.as_color();
                    // If nothing else set base from surface.
                    if result.base_color_tex.is_none() {
                        result.base_color = c;
                    }
                }
                // Volume (socket 1) → density scalar for future use
                if let Some(v) = find_input(graph, id, 1, &cache) {
                    result.volume_density = v.as_scalar();
                }
                // Displacement (socket 2)
                if let Some(d) = find_input(graph, id, 2, &cache) {
                    result.displacement_scale = d.as_scalar();
                }
            }
        }
    }

    result
}

/// Apply evaluation result onto a material and append any new textures to `textures`.
pub fn apply_eval_to_material(
    material: &mut Material,
    textures: &mut Vec<TextureData>,
    eval: GraphEvalResult,
) {
    material.base_color = eval.base_color;
    material.metallic = eval.metallic;
    material.roughness = eval.roughness;
    material.emissive = eval.emissive;
    material.normal_scale = eval.normal_scale;
    material.occlusion_strength = eval.occlusion_strength;

    let base = textures.len();
    for t in eval.new_textures {
        textures.push(t);
    }
    let map_idx = |idx: Option<usize>| -> Option<usize> {
        idx.map(|i| if i >= base { i } else { i })
    };
    // Indices in eval already account for append order (textures.len() at creation time).
    if let Some(i) = eval.base_color_tex {
        material.base_color_tex = Some(i);
    }
    if let Some(i) = eval.metallic_roughness_tex {
        material.metallic_roughness_tex = Some(i);
    }
    if let Some(i) = eval.normal_tex {
        material.normal_tex = Some(i);
    }
    if let Some(i) = eval.occlusion_tex {
        material.occlusion_tex = Some(i);
    }
    if let Some(i) = eval.emissive_tex {
        material.emissive_tex = Some(i);
    }
    let _ = (map_idx, base);
}

/// Bake the graph's color output into a new RGBA texture of `size`×`size`.
pub fn bake_graph_texture(
    graph: &ShaderGraph,
    material: &Material,
    textures: &[TextureData],
    size: u32,
) -> TextureData {
    let size = size.clamp(64, 2048);
    let mut rgba = vec![0u8; (size * size * 4) as usize];
    let node_map: HashMap<u64, &ShaderNode> = graph.nodes.iter().map(|n| (n.id, n)).collect();
    // Pre-evaluate non-UV-dependent parts once.
    let base_eval = evaluate_graph(graph, material, textures, size);

    for y in 0..size {
        for x in 0..size {
            let u = (x as f32 + 0.5) / size as f32;
            let v = (y as f32 + 0.5) / size as f32;
            let color = sample_graph_at(graph, &node_map, textures, &base_eval, u, v);
            let i = ((y * size + x) * 4) as usize;
            rgba[i] = (color[0].clamp(0.0, 1.0) * 255.0) as u8;
            rgba[i + 1] = (color[1].clamp(0.0, 1.0) * 255.0) as u8;
            rgba[i + 2] = (color[2].clamp(0.0, 1.0) * 255.0) as u8;
            rgba[i + 3] = (color[3].clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    TextureData {
        name: format!("Bake_{size}"),
        width: size,
        height: size,
        rgba8: rgba,
    }
}

fn sample_graph_at(
    graph: &ShaderGraph,
    node_map: &HashMap<u64, &ShaderNode>,
    textures: &[TextureData],
    base: &GraphEvalResult,
    u: f32,
    v: f32,
) -> [f32; 4] {
    // Walk from Image/Noise nodes through Mix/ColorRamp toward Principled/Output.
    // Lightweight per-pixel path for common texture chains.
    let order = topo_order(graph);
    let mut cache: HashMap<(u64, u8), SocketVal> = HashMap::new();
    for id in order {
        let Some(node) = node_map.get(&id) else { continue };
        match node.kind {
            ShaderNodeKind::TexCoord => {
                cache.insert((id, 0), SocketVal::Vector([u, v, 0.0]));
            }
            ShaderNodeKind::Mapping => {
                let inp = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Vector([u, v, 0.0]))
                    .as_vector();
                let s = if node.value.abs() < 1e-6 { 1.0 } else { node.value };
                cache.insert((id, 0), SocketVal::Vector([inp[0] * s, inp[1] * s, inp[2] * s]));
            }
            ShaderNodeKind::ImageTexture => {
                let uv = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Vector([u, v, 0.0]))
                    .as_vector();
                let c = if let Some(ti) = node.image_index.filter(|i| *i < textures.len()) {
                    sample_tex(&textures[ti], uv[0], uv[1])
                } else {
                    [1.0, 1.0, 1.0, 1.0]
                };
                cache.insert((id, 0), SocketVal::Color(c));
            }
            ShaderNodeKind::Noise => {
                let uv = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Vector([u, v, 0.0]))
                    .as_vector();
                let scale = if node.value.abs() < 1e-4 { 5.0 } else { node.value };
                let n = fbm(uv[0] * scale, uv[1] * scale, 5);
                cache.insert((id, 0), SocketVal::Color([n, n, n, 1.0]));
            }
            ShaderNodeKind::Voronoi => {
                let uv = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Vector([u, v, 0.0]))
                    .as_vector();
                let scale = if node.value.abs() < 1e-4 { 5.0 } else { node.value };
                let n = voronoi(uv[0] * scale, uv[1] * scale);
                cache.insert((id, 0), SocketVal::Color([n, n, n, 1.0]));
            }
            ShaderNodeKind::Value => {
                cache.insert((id, 0), SocketVal::Scalar(node.value));
            }
            ShaderNodeKind::Rgb => {
                cache.insert((id, 0), SocketVal::Color(node.color));
            }
            ShaderNodeKind::ColorRamp => {
                let fac = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Scalar(0.5))
                    .as_scalar();
                cache.insert((id, 0), SocketVal::Color(color_ramp(fac)));
            }
            ShaderNodeKind::Mix => {
                let fac = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Scalar(0.5))
                    .as_scalar();
                let a = find_input(graph, id, 1, &cache)
                    .unwrap_or(SocketVal::Color([0.0, 0.0, 0.0, 1.0]))
                    .as_color();
                let b = find_input(graph, id, 2, &cache)
                    .unwrap_or(SocketVal::Color([1.0, 1.0, 1.0, 1.0]))
                    .as_color();
                cache.insert((id, 0), SocketVal::Color(mix4(a, b, fac)));
            }
            ShaderNodeKind::Principled => {
                let c = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Color(base.base_color))
                    .as_color();
                cache.insert((id, 0), SocketVal::Color(c));
            }
            ShaderNodeKind::Diffuse | ShaderNodeKind::Glossy | ShaderNodeKind::Emission => {
                let c = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Color(node.color))
                    .as_color();
                cache.insert((id, 0), SocketVal::Color(c));
            }
            ShaderNodeKind::Output => {
                if let Some(s) = find_input(graph, id, 0, &cache) {
                    cache.insert((id, 0), s);
                }
            }
            _ => {}
        }
    }
    // Prefer Output surface, then Principled, then base eval.
    for n in &graph.nodes {
        if n.kind == ShaderNodeKind::Output {
            if let Some(v) = cache.get(&(n.id, 0)) {
                return v.as_color();
            }
        }
    }
    for n in &graph.nodes {
        if n.kind == ShaderNodeKind::Principled {
            if let Some(v) = cache.get(&(n.id, 0)) {
                return v.as_color();
            }
        }
    }
    base.base_color
}
