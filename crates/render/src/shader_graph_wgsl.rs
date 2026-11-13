use forge_core::shader_graph::{ShaderGraph, ShaderLink, ShaderNode, ShaderNodeKind, SocketType};
use std::collections::{HashMap, HashSet};

const NODE_PARAM_STRIDE: usize = 20;
const MAX_GRAPH_NODES: usize = 48;
const GRAPH_PARAM_COUNT: usize = NODE_PARAM_STRIDE * MAX_GRAPH_NODES;

#[derive(Debug)]
pub(super) struct Program {
    pub source: String,
    pub params: Vec<[f32; 4]>,
}

struct Compiler<'a> {
    nodes: HashMap<u64, &'a ShaderNode>,
    node_slots: HashMap<u64, usize>,
    params: Vec<[f32; 4]>,
    links: HashMap<(u64, u8), &'a ShaderLink>,
    used_outputs: HashSet<(u64, u8)>,
    emitted: HashSet<u64>,
    visiting: HashSet<u64>,
    body: String,
}

pub(super) fn compile(graph: &ShaderGraph) -> Result<Program, String> {
    if graph.nodes.len() > MAX_GRAPH_NODES {
        return Err(format!(
            "graph exceeds the {MAX_GRAPH_NODES}-node real-time compilation limit"
        ));
    }
    let mut nodes = HashMap::with_capacity(graph.nodes.len());
    let mut node_slots = HashMap::with_capacity(graph.nodes.len());
    let mut params = vec![[0.0; 4]; GRAPH_PARAM_COUNT];
    for node in &graph.nodes {
        if nodes.insert(node.id, node).is_some() {
            return Err(format!("graph contains duplicate node id {}", node.id));
        }
        let slot = node_slots.len();
        node_slots.insert(node.id, slot);
        pack_node_params(node, slot * NODE_PARAM_STRIDE, &mut params)?;
    }
    let output = graph
        .nodes
        .iter()
        .find(|node| node.kind == ShaderNodeKind::Output)
        .ok_or_else(|| "graph has no Material Output node".to_string())?;
    let link = graph
        .links
        .iter()
        .find(|link| link.to == output.id && link.to_socket == 0)
        .ok_or_else(|| "Material Output Surface is not connected".to_string())?;

    let mut links = HashMap::with_capacity(graph.links.len());
    for link in &graph.links {
        links.insert((link.to, link.to_socket), link);
    }
    let mut compiler = Compiler {
        nodes,
        node_slots,
        params,
        links,
        used_outputs: graph
            .links
            .iter()
            .map(|link| (link.from, link.from_socket))
            .collect(),
        emitted: HashSet::new(),
        visiting: HashSet::new(),
        body: String::new(),
    };
    let surface = compiler.output(link.from, link.from_socket)?;
    if surface.1 != SocketType::Shader {
        return Err("Material Output Surface must be connected to a shader node".into());
    }
    let source = format!(
        r#"
@group(1) @binding(7) var<uniform> graph_params: array<vec4<f32>, {GRAPH_PARAM_COUNT}>;

fn graph_hash(p: vec3<f32>) -> f32 {{
    let cell = vec3<i32>(floor(p));
    var hash = bitcast<u32>(cell.x) * 374761393u
        + bitcast<u32>(cell.y) * 668265263u
        + bitcast<u32>(cell.z) * 2246822519u;
    hash = (hash ^ (hash >> 13u)) * 1274126177u;
    hash = hash ^ (hash >> 16u);
    return f32(hash & 16777215u) / 16777216.0;
}}

fn graph_white_hash(p: vec3<f32>) -> f32 {{
    let bits = bitcast<vec3<u32>>(p);
    var hash = bits.x * 374761393u
        + bits.y * 668265263u
        + bits.z * 2246822519u;
    hash = (hash ^ (hash >> 13u)) * 1274126177u;
    hash = hash ^ (hash >> 16u);
    return f32(hash & 16777215u) / 16777216.0;
}}

fn graph_white_hash1(p: f32) -> f32 {{
    var hash = bitcast<u32>(p) * 374761393u;
    hash = (hash ^ (hash >> 13u)) * 1274126177u;
    hash = hash ^ (hash >> 16u);
    return f32(hash & 16777215u) / 16777216.0;
}}

fn graph_white_hash4(p: vec4<f32>) -> f32 {{
    let bits = bitcast<vec4<u32>>(p);
    var hash = bits.x * 374761393u
        + bits.y * 668265263u
        + bits.z * 2246822519u
        + bits.w * 3266489917u;
    hash = (hash ^ (hash >> 13u)) * 1274126177u;
    hash = hash ^ (hash >> 16u);
    return f32(hash & 16777215u) / 16777216.0;
}}

fn graph_noise(p: vec3<f32>) -> f32 {{
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (vec3<f32>(3.0) - 2.0 * f);
    let a = graph_hash(i);
    let b = graph_hash(i + vec3<f32>(1.0, 0.0, 0.0));
    let c = graph_hash(i + vec3<f32>(0.0, 1.0, 0.0));
    let d = graph_hash(i + vec3<f32>(1.0, 1.0, 0.0));
    let e = graph_hash(i + vec3<f32>(0.0, 0.0, 1.0));
    let f0 = graph_hash(i + vec3<f32>(1.0, 0.0, 1.0));
    let g = graph_hash(i + vec3<f32>(0.0, 1.0, 1.0));
    let h = graph_hash(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(a, b, u.x);
    let x10 = mix(c, d, u.x);
    let x01 = mix(e, f0, u.x);
    let x11 = mix(g, h, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}}

fn graph_hash4(p: vec4<f32>) -> f32 {{
    let cell = vec4<i32>(floor(p));
    var hash = bitcast<u32>(cell.x) * 374761393u
        + bitcast<u32>(cell.y) * 668265263u
        + bitcast<u32>(cell.z) * 2246822519u
        + bitcast<u32>(cell.w) * 3266489917u;
    hash = (hash ^ (hash >> 13u)) * 1274126177u;
    hash = hash ^ (hash >> 16u);
    return f32(hash & 16777215u) / 16777216.0;
}}

fn graph_noise4(p: vec4<f32>) -> f32 {{
    let cell = floor(p);
    let f = fract(p);
    let u = f * f * (vec4<f32>(3.0) - 2.0 * f);
    var total = 0.0;
    for (var w = 0u; w < 2u; w = w + 1u) {{
        for (var z = 0u; z < 2u; z = z + 1u) {{
            for (var y = 0u; y < 2u; y = y + 1u) {{
                for (var x = 0u; x < 2u; x = x + 1u) {{
                    let offset = vec4<f32>(f32(x), f32(y), f32(z), f32(w));
                    let weight = select(1.0 - u.x, u.x, x == 1u)
                        * select(1.0 - u.y, u.y, y == 1u)
                        * select(1.0 - u.z, u.z, z == 1u)
                        * select(1.0 - u.w, u.w, w == 1u);
                    total = total + graph_hash4(cell + offset) * weight;
                }}
            }}
        }}
    }}
    return total;
}}

fn graph_fbm(p: vec3<f32>, detail: f32, roughness: f32, lacunarity: f32, distortion: f32) -> f32 {{
    var q = p;
    if (abs(distortion) > 0.000001) {{
        q = q + vec3<f32>(
            graph_noise(p + vec3<f32>(17.1, 9.2, 3.4)),
            graph_noise(p + vec3<f32>(3.7, 28.4, 11.6)),
            graph_noise(p + vec3<f32>(13.8, 5.2, 24.7))
        ) * distortion;
    }}
    var amplitude = 1.0;
    var frequency = 1.0;
    var total = 0.0;
    var weight = 0.0;
    for (var octave = 0u; octave < 16u; octave = octave + 1u) {{
        let remaining = detail - f32(octave);
        if (remaining <= 0.0) {{ break; }}
        let octave_weight = min(remaining, 1.0);
        total = total + (graph_noise(q * frequency) * 2.0 - 1.0) * amplitude * octave_weight;
        weight = weight + amplitude * octave_weight;
        amplitude = amplitude * roughness;
        frequency = frequency * lacunarity;
    }}
    return clamp(0.5 + 0.5 * total / max(weight, 0.000001), 0.0, 1.0);
}}

fn graph_fbm4(p: vec4<f32>, detail: f32, roughness: f32, lacunarity: f32, distortion: f32) -> f32 {{
    var q = p;
    if (abs(distortion) > 0.000001) {{
        q = q + vec4<f32>(
            graph_noise4(p + vec4<f32>(17.1, 9.2, 3.4, 21.7)),
            graph_noise4(p + vec4<f32>(3.7, 28.4, 11.6, 8.3)),
            graph_noise4(p + vec4<f32>(13.8, 5.2, 24.7, 31.9)),
            graph_noise4(p + vec4<f32>(23.4, 15.6, 7.8, 41.3))
        ) * distortion;
    }}
    var amplitude = 1.0;
    var frequency = 1.0;
    var total = 0.0;
    var weight = 0.0;
    for (var octave = 0u; octave < 16u; octave = octave + 1u) {{
        let remaining = detail - f32(octave);
        if (remaining <= 0.0) {{ break; }}
        let octave_weight = min(remaining, 1.0);
        total = total + (graph_noise4(q * frequency) * 2.0 - 1.0) * amplitude * octave_weight;
        weight = weight + amplitude * octave_weight;
        amplitude = amplitude * roughness;
        frequency = frequency * lacunarity;
    }}
    return clamp(0.5 + 0.5 * total / max(weight, 0.000001), 0.0, 1.0);
}}

fn graph_brick_noise(row: i32, brick: i32) -> f32 {{
    var n = (bitcast<u32>(row) << 16u) + (bitcast<u32>(brick) & 65535u);
    n = (n + 1013u) & 2147483647u;
    n = (n >> 13u) ^ n;
    let nn = (n * (n * n * 60493u + 19990303u) + 1376312589u) & 2147483647u;
    return 0.5 * (f32(nn) / 1073741824.0);
}}

struct GraphBrickSample {{
    color: vec4<f32>,
    mortar: f32,
}}

fn graph_brick(
    vector: vec3<f32>,
    scale: f32,
    color1: vec4<f32>,
    color2: vec4<f32>,
    mortar_color: vec4<f32>,
    mortar_size: f32,
    mortar_smooth: f32,
    bias: f32,
    brick_width: f32,
    row_height: f32,
    offset_amount: f32,
    offset_frequency: i32,
    squash_amount: f32,
    squash_frequency: i32,
) -> GraphBrickSample {{
    let p = vector * scale;
    let height = max(row_height, 0.00001);
    let row = i32(floor(p.y / height));
    var width = max(brick_width, 0.00001);
    let safe_offset_frequency = max(offset_frequency, 1);
    let safe_squash_frequency = max(squash_frequency, 1);
    if (row % safe_squash_frequency == 0) {{
        width = width * max(squash_amount, 0.0001);
    }}
    var offset = 0.0;
    if (row % safe_offset_frequency == 0) {{
        offset = width * offset_amount;
    }}
    let x = p.x + offset;
    let brick = i32(floor(x / width));
    let local_x = x - width * f32(brick);
    let local_y = p.y - height * f32(row);
    let edge = min(min(local_x, local_y), min(width - local_x, height - local_y));
    let size = max(mortar_size, 0.0);
    var mortar = 0.0;
    if (edge < size) {{
        if (mortar_smooth <= 0.0 || size <= 0.0) {{
            mortar = 1.0;
        }} else {{
            let t = clamp((1.0 - edge / size) / mortar_smooth, 0.0, 1.0);
            mortar = t * t * (3.0 - 2.0 * t);
        }}
    }}
    let tint = clamp(graph_brick_noise(row, brick) + bias, 0.0, 1.0);
    let brick_color = mix(color1, color2, tint);
    return GraphBrickSample(mix(brick_color, mortar_color, mortar), mortar);
}}

fn graph_wave(
    vector: vec3<f32>,
    scale: f32,
    distortion: f32,
    detail: f32,
    detail_scale: f32,
    detail_roughness: f32,
    phase: f32,
    wave_type: u32,
    bands_direction: u32,
    rings_direction: u32,
    profile: u32,
) -> f32 {{
    let p = (vector * scale + vec3<f32>(0.000001)) * 0.999999;
    var value = 0.0;
    if (wave_type == 0u) {{
        switch bands_direction {{
            case 1u: {{ value = p.y * 20.0; }}
            case 2u: {{ value = p.z * 20.0; }}
            case 3u: {{ value = (p.x + p.y + p.z) * 10.0; }}
            default: {{ value = p.x * 20.0; }}
        }}
    }} else {{
        var rings = p;
        switch rings_direction {{
            case 0u: {{ rings.x = 0.0; }}
            case 1u: {{ rings.y = 0.0; }}
            case 2u: {{ rings.z = 0.0; }}
            default: {{}}
        }}
        value = length(rings) * 20.0;
    }}
    value = value + phase;
    if (distortion != 0.0) {{
        let noise = graph_fbm(p * detail_scale, clamp(detail, 0.0, 15.0), clamp(detail_roughness, 0.0, 1.0), 2.0, 0.0);
        value = value + distortion * (noise * 2.0 - 1.0);
    }}
    if (profile == 1u) {{
        value = fract(value / 6.28318530718);
    }} else if (profile == 2u) {{
        let normalized = value / 6.28318530718;
        value = abs(normalized - floor(normalized + 0.5)) * 2.0;
    }} else {{
        value = 0.5 + 0.5 * sin(value - 1.57079632679);
    }}
    return value;
}}

fn graph_magic(vector: vec3<f32>, scale: f32, distortion: f32, depth: u32) -> vec4<f32> {{
    let co = vector * scale;
    var x = sin((co.x + co.y + co.z) * 5.0);
    var y = cos((-co.x + co.y - co.z) * 5.0);
    var z = -cos((-co.x - co.y + co.z) * 5.0);
    if (depth > 0u) {{
        x = x * distortion;
        y = y * distortion;
        z = z * distortion;
        y = -cos(x - y + z) * distortion;
        if (depth > 1u) {{
            x = cos(x - y - z) * distortion;
            if (depth > 2u) {{
                z = sin(-x - y - z) * distortion;
                if (depth > 3u) {{
                    x = -cos(-x + y - z) * distortion;
                    if (depth > 4u) {{
                        y = -sin(-x + y + z) * distortion;
                        if (depth > 5u) {{
                            y = -cos(-x + y + z) * distortion;
                            if (depth > 6u) {{
                                x = cos(x + y + z) * distortion;
                                if (depth > 7u) {{
                                    z = sin(x + y - z) * distortion;
                                    if (depth > 8u) {{
                                        x = -cos(-x - y + z) * distortion;
                                        if (depth > 9u) {{
                                            y = -sin(x - y + z) * distortion;
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
    if (distortion != 0.0) {{
        let divisor = distortion * 2.0;
        x = x / divisor;
        y = y / divisor;
        z = z / divisor;
    }}
    return vec4<f32>(0.5 - x, 0.5 - y, 0.5 - z, 1.0);
}}

struct GraphVoronoiSample {{
    distance: f32,
    color: vec3<f32>,
    position: vec3<f32>,
    radius: f32,
}}

fn graph_voronoi(p: vec3<f32>, randomness: f32, smoothness: f32, mode: u32) -> GraphVoronoiSample {{
    let cell = floor(p);
    var first = 1000000.0;
    var second = 1000000.0;
    var first_position = vec3<f32>(0.0);
    var second_position = vec3<f32>(0.0);
    var first_color = vec3<f32>(0.5);
    var second_color = vec3<f32>(0.5);
    for (var z = -1; z <= 1; z = z + 1) {{
        for (var y = -1; y <= 1; y = y + 1) {{
            for (var x = -1; x <= 1; x = x + 1) {{
                let offset = vec3<f32>(f32(x), f32(y), f32(z));
                let current_cell = cell + offset;
                let jitter = vec3<f32>(
                    graph_hash(current_cell + vec3<f32>(17.0, 29.0, 43.0)),
                    graph_hash(current_cell + vec3<f32>(53.0, 71.0, 97.0)),
                    graph_hash(current_cell + vec3<f32>(101.0, 131.0, 157.0))
                );
                let feature = current_cell + vec3<f32>(0.5) + (jitter - vec3<f32>(0.5)) * clamp(randomness, 0.0, 1.0);
                let distance = length(p - feature);
                let current_color = vec3<f32>(
                    graph_hash(current_cell + vec3<f32>(181.0, 191.0, 199.0)),
                    graph_hash(current_cell + vec3<f32>(211.0, 223.0, 227.0)),
                    graph_hash(current_cell + vec3<f32>(229.0, 233.0, 239.0))
                );
                if (distance < first) {{
                    second = first;
                    second_position = first_position;
                    second_color = first_color;
                    first = distance;
                    first_position = feature;
                    first_color = current_color;
                }} else if (distance < second) {{
                    second = distance;
                    second_position = feature;
                    second_color = current_color;
                }}
            }}
        }}
    }}
    let separation = max(length(second_position - first_position), 0.00001);
    let smooth_width = max(smoothness, 0.00001);
    let smooth_weight = clamp(0.5 + 0.5 * (second - first) / smooth_width, 0.0, 1.0);
    let smooth_distance = mix(second, first, smooth_weight) - smooth_width * smooth_weight * (1.0 - smooth_weight);
    let smooth_color = mix(second_color, first_color, smooth_weight);
    let smooth_position = mix(second_position, first_position, smooth_weight);
    let n_sphere_radius = separation * 0.5;
    var value = first;
    var color = first_color;
    var position = first_position;
    switch mode {{
        case 1u: {{ value = second; color = second_color; position = second_position; }}
        case 2u: {{ value = max(smooth_distance, 0.0); color = smooth_color; position = smooth_position; }}
        case 3u: {{
            value = max((second * second - first * first) / (2.0 * separation), 0.0);
        }}
        default: {{}}
    }}
    return GraphVoronoiSample(clamp(value, 0.0, 1.0), color, position, n_sphere_radius);
}}

fn graph_gabor(p: vec3<f32>, frequency: f32, anisotropy: f32, orientation: f32, three_dimensional: bool) -> f32 {{
    let cell = floor(p);
    let sine = sin(orientation);
    let cosine = cos(orientation);
    var total = 0.0;
    var weight = 0.0;
    for (var z = -1; z <= 1; z = z + 1) {{
        for (var y = -1; y <= 1; y = y + 1) {{
            for (var x = -1; x <= 1; x = x + 1) {{
                if (three_dimensional || z == 0) {{
                    let offset = vec3<f32>(f32(x), f32(y), f32(z));
                    let feature_cell = cell + offset;
                    let feature = feature_cell + vec3<f32>(
                        graph_hash(feature_cell + vec3<f32>(13.0, 37.0, 59.0)),
                        graph_hash(feature_cell + vec3<f32>(73.0, 89.0, 107.0)),
                        graph_hash(feature_cell + vec3<f32>(127.0, 149.0, 163.0))
                    );
                    let delta = p - feature;
                    let along = delta.x * cosine + delta.y * sine;
                    let across = -delta.x * sine + delta.y * cosine;
                    let aniso = max(anisotropy, 0.01);
                    var radius = along * along + across * across * aniso;
                    if (three_dimensional) {{
                        radius = radius + delta.z * delta.z * aniso;
                    }}
                    let envelope = exp(-radius);
                    let phase = graph_hash(feature_cell + vec3<f32>(173.0, 197.0, 223.0)) * 6.28318530718;
                    total = total + envelope * cos(along * frequency * 6.28318530718 + phase);
                    weight = weight + envelope;
                }}
            }}
        }}
    }}
    return clamp(0.5 + 0.5 * total / max(weight, 0.000001), 0.0, 1.0);
}}

fn graph_math(mode: u32, a: f32, b: f32) -> f32 {{
    switch mode {{
        case 0u: {{ return a + b; }}
        case 1u: {{ return a - b; }}
        case 2u: {{ return a * b; }}
        case 3u: {{ return select(0.0, a / b, abs(b) > 0.00000001); }}
        case 4u: {{ return pow(max(a, 0.0), b); }}
        case 5u: {{ return select(0.0, log(a) / log(b), a > 0.0 && b > 0.0 && b != 1.0); }}
        case 6u: {{ return sqrt(max(a, 0.0)); }}
        case 7u: {{ return inverseSqrt(max(a, 0.00000001)); }}
        case 8u: {{ return abs(a); }}
        case 9u: {{ return exp(a); }}
        case 10u: {{ return min(a, b); }}
        case 11u: {{ return max(a, b); }}
        case 12u: {{ return select(0.0, 1.0, a < b); }}
        case 13u: {{ return select(0.0, 1.0, a > b); }}
        case 14u: {{ return sign(a); }}
        case 15u: {{ return select(0.0, 1.0, abs(a - b) < 0.00001); }}
        case 16u: {{
            let k = max(b, 0.00001);
            let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
            return a * h + b * (1.0 - h) - k * h * (1.0 - h);
        }}
        case 17u: {{
            let k = max(b, 0.00001);
            let h = clamp(0.5 + 0.5 * (a - b) / k, 0.0, 1.0);
            return a * h + b * (1.0 - h) + k * h * (1.0 - h);
        }}
        case 18u: {{ return round(a); }}
        case 19u: {{ return floor(a); }}
        case 20u: {{ return ceil(a); }}
        case 21u: {{ return trunc(a); }}
        case 22u: {{ return fract(a); }}
        case 23u: {{ return select(0.0, a % b, abs(b) > 0.00000001); }}
        case 24u: {{ return select(a, a - b * floor(a / b), abs(b) > 0.00000001); }}
        case 25u: {{ return select(a, floor(a / b) * b, abs(b) > 0.00000001); }}
        case 26u: {{
            let r = max(abs(b), 0.00000001);
            let t = a - floor(a / (r * 2.0)) * (r * 2.0);
            return select(r * 2.0 - t, t, t < r);
        }}
        case 27u: {{ return sin(a); }}
        case 28u: {{ return cos(a); }}
        case 29u: {{ return tan(a); }}
        case 30u: {{ return asin(clamp(a, -1.0, 1.0)); }}
        case 31u: {{ return acos(clamp(a, -1.0, 1.0)); }}
        case 32u: {{ return atan(a); }}
        case 33u: {{ return atan2(a, b); }}
        case 34u: {{ return sinh(a); }}
        case 35u: {{ return cosh(a); }}
        case 36u: {{ return tanh(a); }}
        case 37u: {{ return a * 0.017453292519943295; }}
        case 38u: {{ return a * 57.29577951308232; }}
        default: {{ return a + b; }}
    }}
}}

fn graph_blend(mode: u32, a: f32, b: f32) -> f32 {{
    switch mode {{
        case 1u: {{ return min(a, b); }}
        case 2u: {{ return a * b; }}
        case 3u: {{ return max(a, b); }}
        case 4u: {{ return a + b - a * b; }}
        case 5u: {{ return select(1.0 - 2.0 * (1.0 - a) * (1.0 - b), 2.0 * a * b, a <= 0.5); }}
        case 6u: {{
            if (b <= 0.5) {{ return a - (1.0 - 2.0 * b) * a * (1.0 - a); }}
            return a + (2.0 * b - 1.0) * (sqrt(max(a, 0.0)) - a);
        }}
        case 7u: {{ return clamp(a + 2.0 * b - 1.0, 0.0, 1.0); }}
        case 8u: {{ return abs(a - b); }}
        case 9u: {{ return a + b - 2.0 * a * b; }}
        case 10u: {{ return a + b; }}
        case 11u: {{ return a - b; }}
        case 12u: {{ return select(0.0, a / b, abs(b) > 0.00000001); }}
        default: {{ return b; }}
    }}
}}

fn graph_surface(in: VsOut) -> GraphSurface {{
{}
    return {};
}}
"#,
        compiler.body, surface.0
    );
    Ok(Program {
        source,
        params: compiler.params,
    })
}

pub(super) fn shader_source(
    graph: &ShaderGraph,
    common: &str,
    mesh: &str,
) -> Result<Program, String> {
    let mut program = compile(graph)?;
    let evaluator = std::mem::take(&mut program.source);
    let start_marker = "// GRAPH_EVALUATOR_BEGIN";
    let end_marker = "// GRAPH_EVALUATOR_END";
    let start = mesh
        .find(start_marker)
        .ok_or_else(|| "mesh shader is missing its graph evaluator start marker".to_string())?;
    let end = mesh
        .find(end_marker)
        .ok_or_else(|| "mesh shader is missing its graph evaluator end marker".to_string())?;
    if end <= start {
        return Err("mesh shader graph evaluator markers are out of order".into());
    }
    let mut mesh = mesh.to_string();
    let end = mesh.find(end_marker).unwrap() + end_marker.len();
    mesh.replace_range(start..end, &evaluator);
    program.source = format!("{common}\n{mesh}");
    Ok(program)
}

impl Compiler<'_> {
    fn line(&mut self, node_id: u64, socket: usize, ty: &str, expression: String) {
        self.body.push_str(&format!(
            "    let n{node_id}_o{socket}: {ty} = {expression};\n"
        ));
    }

    fn output(&mut self, node_id: u64, socket: u8) -> Result<(String, SocketType), String> {
        let node = *self
            .nodes
            .get(&node_id)
            .ok_or_else(|| format!("graph link references missing node {node_id}"))?;
        if !self.emitted.contains(&node_id) {
            if !self.visiting.insert(node_id) {
                return Err(format!("graph contains a cycle at node {node_id}"));
            }
            self.emit_node(node)?;
            self.visiting.remove(&node_id);
            self.emitted.insert(node_id);
        }
        let output = node
            .kind
            .outputs()
            .get(socket as usize)
            .ok_or_else(|| format!("{} has no output socket {socket}", node.kind.title()))?;
        Ok((format!("n{node_id}_o{socket}"), output.ty))
    }

    fn input(&mut self, node: &ShaderNode, socket: u8) -> Result<String, String> {
        if let Some(link) = self.links.get(&(node.id, socket)) {
            let (name, ty) = self.output(link.from, link.from_socket)?;
            if ty == SocketType::Shader {
                return Ok(name);
            }
            let target = node
                .kind
                .inputs()
                .get(socket as usize)
                .ok_or_else(|| format!("{} has no input socket {socket}", node.kind.title()))?;
            return Ok(match target.ty {
                SocketType::Float => format!("{name}.x"),
                SocketType::Color => name,
                SocketType::Vector => format!("{name}.xyz"),
                SocketType::Shader => {
                    return Err(format!("invalid shader input on {}", node.kind.title()))
                }
                SocketType::String => {
                    return Err(format!(
                        "string input is not supported on {}",
                        node.kind.title()
                    ))
                }
            });
        }
        let value = self
            .node_slots
            .get(&node.id)
            .copied()
            .ok_or_else(|| format!("missing parameter slot for node {}", node.id))?
            * NODE_PARAM_STRIDE
            + 6
            + socket as usize;
        let def = node
            .kind
            .inputs()
            .get(socket as usize)
            .ok_or_else(|| format!("{} has no input socket {socket}", node.kind.title()))?;
        if def.ty == SocketType::Vector && node.uses_implicit_coordinates(socket as usize) {
            return Ok("in.local_pos".into());
        }
        Ok(match def.ty {
            SocketType::Float => format!("graph_params[{value}].x"),
            SocketType::Color => format!("graph_params[{value}]"),
            SocketType::Vector => format!("graph_params[{value}].xyz"),
            SocketType::Shader => {
                return Err(format!("unlinked shader input on {}", node.kind.title()))
            }
            SocketType::String => {
                return Err(format!(
                    "string input is not supported on {}",
                    node.kind.title()
                ))
            }
        })
    }

    fn emit_node(&mut self, node: &ShaderNode) -> Result<(), String> {
        let id = node.id;
        match node.kind {
            ShaderNodeKind::Value => {
                let slot = self.node_slots[&id] * NODE_PARAM_STRIDE + 5;
                self.line(id, 0, "vec4<f32>", format!("graph_params[{slot}]"));
            }
            ShaderNodeKind::Rgb => {
                let slot = self.node_slots[&id] * NODE_PARAM_STRIDE;
                self.line(id, 0, "vec4<f32>", format!("graph_params[{slot}]"));
                self.line(
                    id,
                    1,
                    "vec4<f32>",
                    format!("vec4<f32>(graph_params[{slot}].a, graph_params[{slot}].a, graph_params[{slot}].a, 1.0)"),
                );
            }
            ShaderNodeKind::TexCoord | ShaderNodeKind::UvMap => {
                self.line(id, 0, "vec4<f32>", "vec4<f32>(in.uv, 0.0, 0.0)".into());
            }
            ShaderNodeKind::Mapping => {
                let vector = self.input(node, 0)?;
                let location = self.input(node, 1)?;
                let rotation = self.input(node, 2)?;
                let scale = self.input(node, 3)?;
                let name = format!("mapping_{id}");
                self.body.push_str(&format!(
                    "    let {name}_scaled = {vector} * {scale};\n\
                     \x20   let {name}_cx = cos({rotation}.x);\n\
                     \x20   let {name}_sx = sin({rotation}.x);\n\
                     \x20   let {name}_cy = cos({rotation}.y);\n\
                     \x20   let {name}_sy = sin({rotation}.y);\n\
                     \x20   let {name}_cz = cos({rotation}.z);\n\
                     \x20   let {name}_sz = sin({rotation}.z);\n\
                     \x20   let {name}_rx = vec3<f32>({name}_scaled.x, {name}_scaled.y * {name}_cx - {name}_scaled.z * {name}_sx, {name}_scaled.y * {name}_sx + {name}_scaled.z * {name}_cx);\n\
                     \x20   let {name}_ry = vec3<f32>({name}_rx.x * {name}_cy + {name}_rx.z * {name}_sy, {name}_rx.y, -{name}_rx.x * {name}_sy + {name}_rx.z * {name}_cy);\n\
                     \x20   let {name}_rz = vec3<f32>({name}_ry.x * {name}_cz - {name}_ry.y * {name}_sz, {name}_ry.x * {name}_sz + {name}_ry.y * {name}_cz, {name}_ry.z);\n",
                ));
                self.line(
                    id,
                    0,
                    "vec4<f32>",
                    format!("vec4<f32>({name}_rz + {location}, 0.0)"),
                );
            }
            ShaderNodeKind::Noise => {
                if node.mode > 3 {
                    return Err(format!(
                        "GPU Noise does not support dimension mode {}",
                        node.mode
                    ));
                }
                let vector = self.input(node, 0)?;
                let w = self.input(node, 1)?;
                let scale = self.input(node, 2)?;
                let detail = self.input(node, 3)?;
                let roughness = self.input(node, 4)?;
                let lacunarity = self.input(node, 5)?;
                let distortion = self.input(node, 6)?;
                let vector_3d = match node.mode {
                    0 => format!("vec3<f32>({vector}.x, 0.0, 0.0)"),
                    1 => format!("vec3<f32>({vector}.xy, 0.0)"),
                    _ => vector.clone(),
                };
                let vector_4d = format!("vec4<f32>({vector}, {w}) * {scale}");
                let fbm = |point: &str| {
                    if node.mode == 3 {
                        format!("graph_fbm4({point}, clamp({detail}, 0.0, 15.0), clamp({roughness}, 0.0, 1.0), max({lacunarity}, 1.0), {distortion})")
                    } else {
                        format!("graph_fbm({point}, clamp({detail}, 0.0, 15.0), clamp({roughness}, 0.0, 1.0), max({lacunarity}, 1.0), {distortion})")
                    }
                };
                let point = if node.mode == 3 {
                    vector_4d
                } else {
                    format!("{vector_3d} * {scale}")
                };
                let expr = fbm(&point);
                let fac = format!("noise_{id}_fac");
                self.body.push_str(&format!("    let {fac} = {expr};\n"));
                if self.used_outputs.contains(&(id, 0)) {
                    self.line(
                        id,
                        0,
                        "vec4<f32>",
                        format!("vec4<f32>({fac}, {fac}, {fac}, 1.0)"),
                    );
                }
                if self.used_outputs.contains(&(id, 1)) {
                    let (green_point, blue_point) = if node.mode == 3 {
                        (
                            format!("({point} + vec4<f32>(17.13, 29.71, 47.83, 11.17))"),
                            format!("({point} + vec4<f32>(61.37, 13.91, 83.17, 31.23))"),
                        )
                    } else {
                        (
                            format!("({point} + vec3<f32>(17.13, 29.71, 47.83))"),
                            format!("({point} + vec3<f32>(61.37, 13.91, 83.17))"),
                        )
                    };
                    let green = fbm(&green_point);
                    let blue = fbm(&blue_point);
                    self.line(
                        id,
                        1,
                        "vec4<f32>",
                        format!("vec4<f32>({fac}, {green}, {blue}, 1.0)"),
                    );
                }
            }
            ShaderNodeKind::WhiteNoise => {
                if node.mode > 3 {
                    return Err(format!(
                        "GPU White Noise does not support dimension mode {}",
                        node.mode
                    ));
                }
                let vector = self.input(node, 0)?;
                let w = self.input(node, 1)?;
                let point_3d = match node.mode {
                    0 => format!("vec3<f32>({vector}.x, 0.0, 0.0)"),
                    1 => format!("vec3<f32>({vector}.xy, 0.0)"),
                    _ => vector.clone(),
                };
                let value = if node.mode == 3 {
                    format!("graph_white_hash4(vec4<f32>({vector}, {w}))")
                } else if node.mode == 0 {
                    format!("graph_white_hash1({w})")
                } else {
                    format!("graph_white_hash({point_3d})")
                };
                let value_name = format!("white_noise_{id}_value");
                self.body
                    .push_str(&format!("    let {value_name} = {value};\n"));
                if self.used_outputs.contains(&(id, 0)) {
                    self.line(
                        id,
                        0,
                        "vec4<f32>",
                        format!("vec4<f32>({value_name}, {value_name}, {value_name}, 1.0)"),
                    );
                }
                if self.used_outputs.contains(&(id, 1)) {
                    let (green, blue) = if node.mode == 3 {
                        (
                            format!(
                                "graph_white_hash4(vec4<f32>({vector}, {w}) + vec4<f32>(17.13, 29.71, 47.83, 61.37))"
                            ),
                            format!(
                                "graph_white_hash4(vec4<f32>({vector}, {w}) + vec4<f32>(34.26, 59.42, 95.66, 122.74))"
                            ),
                        )
                    } else if node.mode == 0 {
                        (
                            format!("graph_white_hash1({w} + 61.37)"),
                            format!("graph_white_hash1({w} + 122.74)"),
                        )
                    } else {
                        (
                            format!(
                                "graph_white_hash({point_3d} + vec3<f32>(17.13, 29.71, 47.83))"
                            ),
                            format!(
                                "graph_white_hash({point_3d} + vec3<f32>(61.37, 13.91, 83.17))"
                            ),
                        )
                    };
                    self.line(
                        id,
                        1,
                        "vec4<f32>",
                        format!("vec4<f32>({value_name}, {green}, {blue}, 1.0)"),
                    );
                }
            }
            ShaderNodeKind::Checker => {
                let vector = self.input(node, 0)?;
                let color1 = self.input(node, 1)?;
                let color2 = self.input(node, 2)?;
                let scale = self.input(node, 3)?;
                let fac = format!("checker_{id}_fac");
                self.body.push_str(&format!(
                    "    let {fac} = select(0.0, 1.0, (i32(floor(({vector} * {scale}).x)) + i32(floor(({vector} * {scale}).y)) + i32(floor(({vector} * {scale}).z))) % 2 == 0);\n"
                ));
                if self.used_outputs.contains(&(id, 0)) {
                    self.line(
                        id,
                        0,
                        "vec4<f32>",
                        format!("mix({color2}, {color1}, {fac})"),
                    );
                }
                if self.used_outputs.contains(&(id, 1)) {
                    self.line(
                        id,
                        1,
                        "vec4<f32>",
                        format!("vec4<f32>({fac}, {fac}, {fac}, 1.0)"),
                    );
                }
            }
            ShaderNodeKind::Voronoi => {
                let vector = self.input(node, 0)?;
                let scale = self.input(node, 2)?;
                let smoothness = self.input(node, 6)?;
                let randomness = self.input(node, 7)?;
                let sample = format!("voronoi_{id}");
                self.body.push_str(&format!(
                    "    let {sample} = graph_voronoi({vector} * {scale}, {randomness}, {smoothness}, {}u);\n",
                    node.mode.min(4)
                ));
                if self.used_outputs.contains(&(id, 0)) {
                    self.line(
                        id,
                        0,
                        "vec4<f32>",
                        format!("vec4<f32>({sample}.distance, {sample}.distance, {sample}.distance, 1.0)"),
                    );
                }
                if self.used_outputs.contains(&(id, 1)) {
                    self.line(
                        id,
                        1,
                        "vec4<f32>",
                        format!("vec4<f32>({sample}.color, 1.0)"),
                    );
                }
                if self.used_outputs.contains(&(id, 2)) {
                    self.line(
                        id,
                        2,
                        "vec4<f32>",
                        format!("vec4<f32>({sample}.position, 1.0)"),
                    );
                }
                if self.used_outputs.contains(&(id, 3)) {
                    self.line(
                        id,
                        3,
                        "vec4<f32>",
                        format!("vec4<f32>({sample}.position.z, 0.0, 0.0, 1.0)"),
                    );
                }
                if self.used_outputs.contains(&(id, 4)) {
                    self.line(
                        id,
                        4,
                        "vec4<f32>",
                        format!("vec4<f32>({sample}.radius, 0.0, 0.0, 1.0)"),
                    );
                }
            }
            ShaderNodeKind::Brick => {
                let vector = self.input(node, 0)?;
                let color1 = self.input(node, 1)?;
                let color2 = self.input(node, 2)?;
                let mortar = self.input(node, 3)?;
                let scale = self.input(node, 4)?;
                let mortar_size = self.input(node, 5)?;
                let mortar_smooth = self.input(node, 6)?;
                let bias = self.input(node, 7)?;
                let brick_width = self.input(node, 8)?;
                let row_height = self.input(node, 9)?;
                let base = self.node_slots[&id] * NODE_PARAM_STRIDE;
                let controls = format!("graph_params[{}]", base + 1);
                let name = format!("brick_{id}");
                self.body.push_str(&format!(
                    "    let {name} = graph_brick({vector}, {scale}, {color1}, {color2}, {mortar}, {mortar_size}, {mortar_smooth}, {bias}, {brick_width}, {row_height}, {controls}.x, i32(round({controls}.y)), {controls}.z, i32(round({controls}.w)));\n"
                ));
                if self.used_outputs.contains(&(id, 0)) {
                    self.line(id, 0, "vec4<f32>", format!("{name}.color"));
                }
                if self.used_outputs.contains(&(id, 1)) {
                    self.line(
                        id,
                        1,
                        "vec4<f32>",
                        format!("vec4<f32>({name}.mortar, {name}.mortar, {name}.mortar, 1.0)"),
                    );
                }
            }
            ShaderNodeKind::Gradient => {
                let vector = self.input(node, 0)?;
                let (x, y) = (format!("{vector}.x"), format!("{vector}.y"));
                let dx = format!("({x} - 0.5)");
                let dy = format!("({y} - 0.5)");
                let value = match node.mode.min(6) {
                    1 => format!("clamp({x}, 0.0, 1.0) * clamp({x}, 0.0, 1.0)"),
                    2 => format!("smoothstep(0.0, 1.0, clamp({x}, 0.0, 1.0))"),
                    3 => format!("clamp(({x} + {y}) * 0.5, 0.0, 1.0)"),
                    4 => format!("clamp(1.0 - length(vec2<f32>({dx}, {dy})) * 2.0, 0.0, 1.0)"),
                    5 => format!("clamp(1.0 - dot(vec2<f32>({dx}, {dy}), vec2<f32>({dx}, {dy})) * 4.0, 0.0, 1.0)"),
                    6 => format!("fract(atan2({dy}, {dx}) / 6.28318530718 + 0.5)"),
                    _ => format!("clamp({x}, 0.0, 1.0)"),
                };
                let fac = format!("gradient_{id}_fac");
                self.body.push_str(&format!("    let {fac} = {value};\n"));
                if self.used_outputs.contains(&(id, 0)) {
                    self.line(
                        id,
                        0,
                        "vec4<f32>",
                        format!("vec4<f32>({fac}, {fac}, {fac}, 1.0)"),
                    );
                }
                if self.used_outputs.contains(&(id, 1)) {
                    self.line(
                        id,
                        1,
                        "vec4<f32>",
                        format!("vec4<f32>({fac}, {fac}, {fac}, 1.0)"),
                    );
                }
            }
            ShaderNodeKind::Magic => {
                let vector = self.input(node, 0)?;
                let scale = self.input(node, 1)?;
                let distortion = self.input(node, 2)?;
                let name = format!("magic_{id}");
                let base = self.node_slots[&id] * NODE_PARAM_STRIDE;
                let properties = format!("graph_params[{}]", base + 3);
                let depth = if node.props[9] > 0.5 {
                    format!("u32(round({properties}.x))")
                } else {
                    format!("{}u", node.mode.saturating_add(2).min(10))
                };
                self.body.push_str(&format!(
                    "    let {name}_color = graph_magic({vector}, {scale}, {distortion}, {depth});\n\
                     \x20   let {name}_fac = dot({name}_color.rgb, vec3<f32>(0.3333333333));\n"
                ));
                if self.used_outputs.contains(&(id, 0)) {
                    self.line(
                        id,
                        0,
                        "vec4<f32>",
                        format!("vec4<f32>({name}_fac, {name}_fac, {name}_fac, 1.0)"),
                    );
                }
                if self.used_outputs.contains(&(id, 1)) {
                    self.line(id, 1, "vec4<f32>", format!("{name}_color"));
                }
            }
            ShaderNodeKind::Gabor => {
                let vector = self.input(node, 0)?;
                let scale = self.input(node, 1)?;
                let frequency = self.input(node, 2)?;
                let anisotropy = self.input(node, 3)?;
                let orientation = self.input(node, 4)?;
                let fac = format!(
                    "graph_gabor({vector} * {scale}, {frequency}, {anisotropy}, {orientation}, {} )",
                    node.mode == 1
                );
                let fac_name = format!("gabor_{id}_fac");
                self.body
                    .push_str(&format!("    let {fac_name} = {fac};\n"));
                if self.used_outputs.contains(&(id, 0)) {
                    self.line(
                        id,
                        0,
                        "vec4<f32>",
                        format!("vec4<f32>({fac_name}, {fac_name}, {fac_name}, 1.0)"),
                    );
                }
                if self.used_outputs.contains(&(id, 1)) {
                    self.line(
                        id,
                        1,
                        "vec4<f32>",
                        format!("vec4<f32>({fac_name}, {fac_name}, {fac_name}, 1.0)"),
                    );
                }
            }
            ShaderNodeKind::Wave => {
                let vector = self.input(node, 0)?;
                let scale = self.input(node, 1)?;
                let distortion = self.input(node, 2)?;
                let detail = self.input(node, 3)?;
                let detail_scale = self.input(node, 4)?;
                let detail_roughness = self.input(node, 5)?;
                let phase = self.input(node, 6)?;
                let base = self.node_slots[&id] * NODE_PARAM_STRIDE;
                let properties = format!("graph_params[{}]", base + 3);
                let fac = format!(
                    "graph_wave({vector}, {scale}, {distortion}, {detail}, {detail_scale}, {detail_roughness}, {phase}, {}u, u32(round({properties}.x)), u32(round({properties}.y)), u32(round({properties}.z)))",
                    node.mode.min(1)
                );
                let fac_name = format!("wave_{id}_fac");
                self.body
                    .push_str(&format!("    let {fac_name} = {fac};\n"));
                if self.used_outputs.contains(&(id, 0)) {
                    self.line(
                        id,
                        0,
                        "vec4<f32>",
                        format!("vec4<f32>({fac_name}, {fac_name}, {fac_name}, 1.0)"),
                    );
                }
                if self.used_outputs.contains(&(id, 1)) {
                    self.line(
                        id,
                        1,
                        "vec4<f32>",
                        format!("vec4<f32>({fac_name}, {fac_name}, {fac_name}, 1.0)"),
                    );
                }
            }
            ShaderNodeKind::Math => {
                let a = self.input(node, 0)?;
                let b = self.input(node, 1)?;
                let value = format!("graph_math({}u, {a}, {b})", node.mode.min(38));
                self.line(
                    id,
                    0,
                    "vec4<f32>",
                    format!("vec4<f32>({value}, {value}, {value}, 1.0)"),
                );
            }
            ShaderNodeKind::ColorRamp => {
                let fac = self.input(node, 0)?;
                let (colors, _, interp, _) = forge_core::shader_graph::ramp_get_stops(node);
                let base = self.node_slots[&id] * NODE_PARAM_STRIDE;
                let color = |index: usize| format!("graph_params[{}]", base + index);
                let position = |index: usize| {
                    let component = ["x", "y", "z", "w"][index];
                    format!("graph_params[{}].{component}", base + 4)
                };
                let mut expression = color(colors.len().saturating_sub(1));
                for index in (0..colors.len().saturating_sub(1)).rev() {
                    let left = color(index);
                    let right = color(index + 1);
                    let start = position(index);
                    let end = position(index + 1);
                    let t = if interp == 2 {
                        "0.0".to_string()
                    } else {
                        let f = format!(
                            "clamp(({fac} - {start}) / max({end} - {start}, 0.000001), 0.0, 1.0)"
                        );
                        if interp == 1 {
                            format!("smoothstep(0.0, 1.0, {f})")
                        } else {
                            f
                        }
                    };
                    expression =
                        format!("select({expression}, mix({left}, {right}, {t}), {fac} < {end})");
                }
                self.line(id, 0, "vec4<f32>", expression);
                self.line(
                    id,
                    1,
                    "vec4<f32>",
                    format!("vec4<f32>(n{id}_o0.a, n{id}_o0.a, n{id}_o0.a, 1.0)"),
                );
            }
            ShaderNodeKind::Mix => {
                let fac = self.input(node, 0)?;
                let a = self.input(node, 1)?;
                let b = self.input(node, 2)?;
                let blended = format!("vec3<f32>(graph_blend({}u, {a}.r, {b}.r), graph_blend({}u, {a}.g, {b}.g), graph_blend({}u, {a}.b, {b}.b))", node.mode.min(12), node.mode.min(12), node.mode.min(12));
                let mixed = format!("vec4<f32>(mix({a}.rgb, {blended}, clamp({fac}, 0.0, 1.0)), mix({a}.a, {b}.a, clamp({fac}, 0.0, 1.0)))");
                self.line(id, 0, "vec4<f32>", mixed);
                self.line(
                    id,
                    1,
                    "vec4<f32>",
                    format!("vec4<f32>(n{id}_o0.a, n{id}_o0.a, n{id}_o0.a, 1.0)"),
                );
            }
            ShaderNodeKind::CombineXyz | ShaderNodeKind::CombineColor => {
                let x = self.input(node, 0)?;
                let y = self.input(node, 1)?;
                let z = self.input(node, 2)?;
                self.line(id, 0, "vec4<f32>", format!("vec4<f32>({x}, {y}, {z}, 1.0)"));
            }
            ShaderNodeKind::SeparateXyz | ShaderNodeKind::SeparateColor => {
                let input = self.input(node, 0)?;
                for (socket, component) in ["x", "y", "z"].iter().enumerate() {
                    let value = format!("{input}.{component}");
                    self.line(
                        id,
                        socket,
                        "vec4<f32>",
                        format!("vec4<f32>({value}, {value}, {value}, 1.0)"),
                    );
                }
            }
            ShaderNodeKind::Principled => {
                let color = self.input(node, 0)?;
                let metallic = self.input(node, 1)?;
                let roughness = self.input(node, 2)?;
                let emission_color = self.input(node, 6)?;
                let emission_strength = self.input(node, 7)?;
                let ao = self.input(node, 13)?;
                let alpha = self.input(node, 4)?;
                let normal = if self.links.contains_key(&(id, 5)) {
                    let normal = self.input(node, 5)?;
                    format!("normalize({normal})")
                } else {
                    "normalize(in.normal)".into()
                };
                self.line(id, 0, "GraphSurface", format!(
                    "GraphSurface(vec4<f32>({color}.rgb, clamp({alpha}, 0.0, 1.0)), clamp({metallic}, 0.0, 1.0), clamp({roughness}, 0.045, 1.0), ({emission_color}.rgb * {emission_strength}), {normal}, clamp({ao}, 0.0, 1.0))"
                ));
            }
            ShaderNodeKind::Emission | ShaderNodeKind::Background => {
                let color = self.input(node, 0)?;
                let strength = self.input(node, 1)?;
                self.line(id, 0, "GraphSurface", format!(
                    "GraphSurface(vec4<f32>(0.0, 0.0, 0.0, 1.0), 0.0, 1.0, {color}.rgb * {strength}, normalize(in.normal), 1.0)"
                ));
            }
            _ => {
                return Err(format!(
                    "{} is not yet supported by the real-time graph compiler",
                    node.kind.title()
                ))
            }
        }
        Ok(())
    }
}

fn pack_node_params(node: &ShaderNode, base: usize, params: &mut [[f32; 4]]) -> Result<(), String> {
    params[base] = node.color;
    for group in 0..4 {
        params[base + 1 + group].copy_from_slice(&node.props[group * 4..group * 4 + 4]);
    }
    if node.kind == ShaderNodeKind::Brick && node.props[..4].iter().all(|value| *value == 0.0) {
        params[base + 1] = [0.5, 2.0, 1.0, 2.0];
    }
    params[base + 5] = [node.value; 4];

    if node.kind == ShaderNodeKind::ColorRamp {
        let (colors, positions, _, _) = forge_core::shader_graph::ramp_get_stops(node);
        for index in 0..4 {
            params[base + index] = colors.get(index).copied().unwrap_or([0.0, 0.0, 0.0, 1.0]);
        }
        let mut ramp_positions = [0.0; 4];
        for (index, position) in positions.into_iter().enumerate().take(4) {
            ramp_positions[index] = position;
        }
        params[base + 4] = ramp_positions;
    }

    let defaults = ShaderNode::with_kind(node.id, node.kind, node.pos);
    for index in 0..node.kind.inputs().len().min(14) {
        let mut value = node
            .input_defaults
            .get(index)
            .copied()
            .or_else(|| defaults.input_defaults.get(index).copied())
            .unwrap_or([0.0; 4]);
        if node.input_defaults.get(index).is_none() {
            match node.kind {
                ShaderNodeKind::Principled => match index {
                    0 => value = node.color,
                    1 => value[0] = node.props[0],
                    2 if node.props[1].abs() > 1e-8 => value[0] = node.props[1],
                    3 if node.props[2].abs() > 1e-8 => value[0] = node.props[2],
                    4 if node.props[3].abs() > 1e-8 => value[0] = node.props[3],
                    _ => {}
                },
                ShaderNodeKind::Math if index == 0 => value[0] = node.value,
                ShaderNodeKind::Math if index == 1 => value[0] = node.props[0],
                ShaderNodeKind::Noise if (3..=6).contains(&index) => {
                    let prop = node.props[index - 3];
                    if prop != 0.0 {
                        value[0] = prop;
                    }
                }
                ShaderNodeKind::Mapping if index == 3 => {
                    value[0] = if (node.value - 0.5).abs() > 1e-8 {
                        node.value
                    } else {
                        1.0
                    };
                    value[1] = if node.props[0].abs() > 1e-8 {
                        node.props[0]
                    } else {
                        1.0
                    };
                    value[2] = if node.props[1].abs() > 1e-8 {
                        node.props[1]
                    } else {
                        1.0
                    };
                }
                ShaderNodeKind::Noise
                | ShaderNodeKind::Voronoi
                | ShaderNodeKind::Wave
                | ShaderNodeKind::Brick
                | ShaderNodeKind::Checker
                | ShaderNodeKind::Magic
                | ShaderNodeKind::Gabor
                    if is_scale_socket(node.kind, index) && (node.value - 0.5).abs() > 1e-8 =>
                {
                    value[0] = node.value;
                }
                _ => {}
            }
        }
        params[base + 6 + index] = value;
    }
    Ok(())
}

fn is_scale_socket(kind: ShaderNodeKind, index: usize) -> bool {
    matches!(
        (kind, index),
        (ShaderNodeKind::Noise | ShaderNodeKind::Voronoi, 2)
            | (
                ShaderNodeKind::Wave | ShaderNodeKind::Magic | ShaderNodeKind::Gabor,
                1
            )
            | (ShaderNodeKind::Brick, 4)
            | (ShaderNodeKind::Checker, 3)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate_graph(graph: &ShaderGraph) {
        let program = shader_source(
            graph,
            include_str!("common.wgsl"),
            include_str!("mesh.wgsl"),
        )
        .expect("graph should compile");
        let source = &program.source;
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("generated graph shader should validate");
    }

    #[test]
    fn default_principled_graph_generates_valid_wgsl() {
        validate_graph(&ShaderGraph::default());
    }

    #[test]
    fn procedural_noise_ramp_chain_generates_valid_wgsl() {
        let graph = ShaderGraph {
            nodes: vec![
                ShaderNode::with_kind(1, ShaderNodeKind::Noise, [0.0, 0.0]),
                ShaderNode::with_kind(2, ShaderNodeKind::ColorRamp, [1.0, 0.0]),
                ShaderNode::with_kind(3, ShaderNodeKind::Principled, [2.0, 0.0]),
                ShaderNode::with_kind(4, ShaderNodeKind::Output, [3.0, 0.0]),
            ],
            links: vec![
                ShaderLink {
                    from: 1,
                    to: 2,
                    from_socket: 0,
                    to_socket: 0,
                },
                ShaderLink {
                    from: 2,
                    to: 3,
                    from_socket: 0,
                    to_socket: 0,
                },
                ShaderLink {
                    from: 3,
                    to: 4,
                    from_socket: 0,
                    to_socket: 0,
                },
            ],
            next_id: 5,
        };
        validate_graph(&graph);
    }

    #[test]
    fn unconnected_procedural_vectors_use_object_space_positions() {
        for (kind, output_socket) in [
            (ShaderNodeKind::Noise, 1),
            (ShaderNodeKind::Checker, 0),
            (ShaderNodeKind::Wave, 1),
        ] {
            let graph = ShaderGraph {
                nodes: vec![
                    ShaderNode::with_kind(1, kind, [0.0, 0.0]),
                    ShaderNode::with_kind(2, ShaderNodeKind::Principled, [1.0, 0.0]),
                    ShaderNode::with_kind(3, ShaderNodeKind::Output, [2.0, 0.0]),
                ],
                links: vec![
                    ShaderLink {
                        from: 1,
                        from_socket: output_socket,
                        to: 2,
                        to_socket: 0,
                    },
                    ShaderLink {
                        from: 2,
                        from_socket: 0,
                        to: 3,
                        to_socket: 0,
                    },
                ],
                next_id: 4,
            };
            validate_graph(&graph);
            let source = compile(&graph)
                .expect("procedural graph should compile")
                .source;
            assert!(
                source.contains("in.local_pos"),
                "{} should use per-fragment object-space positions",
                kind.title()
            );
        }
    }

    #[test]
    fn common_texture_nodes_and_ui_modes_generate_valid_wgsl() {
        let validate_texture = |kind, mode, output_socket| {
            let mut texture = ShaderNode::with_kind(1, kind, [0.0, 0.0]);
            texture.mode = mode;
            let graph = ShaderGraph {
                nodes: vec![
                    texture,
                    ShaderNode::with_kind(2, ShaderNodeKind::Principled, [1.0, 0.0]),
                    ShaderNode::with_kind(3, ShaderNodeKind::Output, [2.0, 0.0]),
                ],
                links: vec![
                    ShaderLink {
                        from: 1,
                        from_socket: output_socket,
                        to: 2,
                        to_socket: 0,
                    },
                    ShaderLink {
                        from: 2,
                        from_socket: 0,
                        to: 3,
                        to_socket: 0,
                    },
                ],
                next_id: 4,
            };
            assert!(super::compile(&graph).is_ok(), "{kind:?} mode {mode}");
            validate_graph(&graph);
        };

        for mode in 0..=3 {
            validate_texture(ShaderNodeKind::Noise, mode, 1);
            validate_texture(ShaderNodeKind::WhiteNoise, mode, 1);
        }
        for mode in 0..=4 {
            validate_texture(ShaderNodeKind::Voronoi, mode, 1);
        }
        for mode in 0..=6 {
            validate_texture(ShaderNodeKind::Gradient, mode, 1);
        }
        for mode in 0..=1 {
            validate_texture(ShaderNodeKind::Wave, mode, 1);
            validate_texture(ShaderNodeKind::Gabor, mode, 1);
        }
        for (kind, output_socket) in [
            (ShaderNodeKind::Brick, 0),
            (ShaderNodeKind::Checker, 0),
            (ShaderNodeKind::Magic, 1),
        ] {
            validate_texture(kind, 0, output_socket);
        }
    }

    #[test]
    fn explicit_texture_coordinate_links_override_implicit_uvs() {
        let graph = ShaderGraph {
            nodes: vec![
                ShaderNode::with_kind(1, ShaderNodeKind::TexCoord, [0.0, 0.0]),
                ShaderNode::with_kind(2, ShaderNodeKind::Noise, [1.0, 0.0]),
                ShaderNode::with_kind(3, ShaderNodeKind::Principled, [2.0, 0.0]),
                ShaderNode::with_kind(4, ShaderNodeKind::Output, [3.0, 0.0]),
            ],
            links: vec![
                ShaderLink {
                    from: 1,
                    from_socket: 0,
                    to: 2,
                    to_socket: 0,
                },
                ShaderLink {
                    from: 2,
                    from_socket: 1,
                    to: 3,
                    to_socket: 0,
                },
                ShaderLink {
                    from: 3,
                    from_socket: 0,
                    to: 4,
                    to_socket: 0,
                },
            ],
            next_id: 5,
        };
        let source = compile(&graph)
            .expect("explicitly mapped graph should compile")
            .source;
        assert!(source.contains("graph_fbm(n1_o0.xyz *"));
    }

    #[test]
    fn noise_fac_does_not_evaluate_unused_color_or_zero_distortion_warp() {
        let graph = ShaderGraph {
            nodes: vec![
                ShaderNode::with_kind(1, ShaderNodeKind::Noise, [0.0, 0.0]),
                ShaderNode::with_kind(2, ShaderNodeKind::ColorRamp, [1.0, 0.0]),
                ShaderNode::with_kind(3, ShaderNodeKind::Principled, [2.0, 0.0]),
                ShaderNode::with_kind(4, ShaderNodeKind::Output, [3.0, 0.0]),
            ],
            links: vec![
                ShaderLink {
                    from: 1,
                    from_socket: 0,
                    to: 2,
                    to_socket: 0,
                },
                ShaderLink {
                    from: 2,
                    from_socket: 0,
                    to: 3,
                    to_socket: 0,
                },
                ShaderLink {
                    from: 3,
                    from_socket: 0,
                    to: 4,
                    to_socket: 0,
                },
            ],
            next_id: 5,
        };
        let program = compile(&graph).expect("Noise Fac graph should compile");
        assert_eq!(
            program.source.matches("graph_fbm(in.local_pos *").count(),
            1
        );
        assert!(program.source.contains("if (abs(distortion) > 0.000001)"));
        validate_graph(&graph);
    }

    #[test]
    fn mapped_texture_mix_math_and_checker_generate_valid_wgsl() {
        let graph = ShaderGraph {
            nodes: vec![
                ShaderNode::with_kind(1, ShaderNodeKind::TexCoord, [0.0, 0.0]),
                ShaderNode::with_kind(2, ShaderNodeKind::Mapping, [1.0, 0.0]),
                ShaderNode::with_kind(3, ShaderNodeKind::Noise, [2.0, 0.0]),
                ShaderNode::with_kind(4, ShaderNodeKind::Wave, [2.0, 1.0]),
                ShaderNode::with_kind(5, ShaderNodeKind::Math, [3.0, 0.0]),
                ShaderNode::with_kind(6, ShaderNodeKind::ColorRamp, [4.0, 0.0]),
                ShaderNode::with_kind(7, ShaderNodeKind::Checker, [3.0, 1.0]),
                ShaderNode::with_kind(8, ShaderNodeKind::Mix, [5.0, 0.0]),
                ShaderNode::with_kind(9, ShaderNodeKind::Principled, [6.0, 0.0]),
                ShaderNode::with_kind(10, ShaderNodeKind::Output, [7.0, 0.0]),
            ],
            links: vec![
                ShaderLink {
                    from: 1,
                    to: 2,
                    from_socket: 0,
                    to_socket: 0,
                },
                ShaderLink {
                    from: 2,
                    to: 3,
                    from_socket: 0,
                    to_socket: 0,
                },
                ShaderLink {
                    from: 2,
                    to: 4,
                    from_socket: 0,
                    to_socket: 0,
                },
                ShaderLink {
                    from: 3,
                    to: 5,
                    from_socket: 0,
                    to_socket: 0,
                },
                ShaderLink {
                    from: 4,
                    to: 5,
                    from_socket: 0,
                    to_socket: 1,
                },
                ShaderLink {
                    from: 3,
                    to: 6,
                    from_socket: 0,
                    to_socket: 0,
                },
                ShaderLink {
                    from: 6,
                    to: 8,
                    from_socket: 0,
                    to_socket: 1,
                },
                ShaderLink {
                    from: 7,
                    to: 8,
                    from_socket: 0,
                    to_socket: 2,
                },
                ShaderLink {
                    from: 5,
                    to: 8,
                    from_socket: 0,
                    to_socket: 0,
                },
                ShaderLink {
                    from: 8,
                    to: 9,
                    from_socket: 0,
                    to_socket: 0,
                },
                ShaderLink {
                    from: 9,
                    to: 10,
                    from_socket: 0,
                    to_socket: 0,
                },
            ],
            next_id: 11,
        };
        validate_graph(&graph);
    }

    #[test]
    fn connected_unsupported_nodes_report_a_fallback_reason() {
        let mut graph = ShaderGraph::default();
        graph.nodes.insert(
            0,
            ShaderNode::with_kind(3, ShaderNodeKind::Diffuse, [1.0, 0.0]),
        );
        graph.links = vec![ShaderLink {
            from: 3,
            to: 2,
            from_socket: 0,
            to_socket: 0,
        }];
        let error = compile(&graph).expect_err("Diffuse is outside the initial GPU subset");
        assert!(error.contains("Diffuse BSDF"));
    }

    #[test]
    fn invalid_noise_dimensions_report_a_fallback_reason() {
        let mut graph = ShaderGraph {
            nodes: vec![
                ShaderNode::with_kind(1, ShaderNodeKind::Noise, [0.0, 0.0]),
                ShaderNode::with_kind(2, ShaderNodeKind::Principled, [1.0, 0.0]),
                ShaderNode::with_kind(3, ShaderNodeKind::Output, [2.0, 0.0]),
            ],
            links: vec![
                ShaderLink {
                    from: 1,
                    to: 2,
                    from_socket: 1,
                    to_socket: 0,
                },
                ShaderLink {
                    from: 2,
                    to: 3,
                    from_socket: 0,
                    to_socket: 0,
                },
            ],
            next_id: 4,
        };
        graph.nodes[0].mode = 4;
        let error =
            compile(&graph).expect_err("unknown Noise dimensions must use the baked fallback");
        assert!(error.contains("dimension mode 4"));
    }

    #[test]
    fn value_edits_update_uniforms_without_changing_shader_source() {
        let graph = ShaderGraph::default();
        let mut edited = graph.clone();
        edited.nodes[0].props[1] = 0.2;
        let first = compile(&graph).expect("default graph should compile");
        let second = compile(&edited).expect("edited graph should compile");
        assert_eq!(first.source, second.source);
        assert_ne!(first.params, second.params);
    }

    #[test]
    fn node_parameters_fit_the_portable_uniform_budget() {
        assert_eq!(
            compile(&ShaderGraph::default()).unwrap().params.len(),
            GRAPH_PARAM_COUNT
        );
        assert_eq!(GRAPH_PARAM_COUNT * std::mem::size_of::<[f32; 4]>(), 15_360);
    }
}
