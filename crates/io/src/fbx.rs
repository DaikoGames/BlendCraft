//! Pure-Rust FBX 7.x import/export — no ufbx / C / online APIs.
//! Works on desktop **and** `wasm32` (browser) via `load_fbx_bytes`.
//!
//! Import: meshes, materials/textures, LimbNode armatures, Skin/Cluster weights
//! (bind poses via Transform / TransformLink), OO connections.
//! Export: static triangle meshes + Phong materials (binary FBX).
use anyhow::{bail, Context, Result};
use forge_core::{AnimationClip, Material, MeshData, MeshGroup, Object, ObjectAnimation, ObjectKind, Scene, SkinBinding, TextureData, Vertex};
use glam::{Mat4, Quat, Vec3};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

/// Looks up an external texture by the path/file name stored in the FBX and returns its bytes.
type TexResolver<'a> = &'a dyn Fn(&str) -> Option<Vec<u8>>;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
pub fn load_fbx(path: &Path) -> Result<Scene> {
    // Pure-Rust FBX parser (no ufbx / C toolchain). Works on desktop and matches
    // the browser path that uses `load_fbx_bytes`.
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let resolver = move |name: &str| crate::find_texture_file(&dir, &stem, name);
    load_fbx_bytes_with(&bytes, &resolver)
}

#[cfg(target_arch = "wasm32")]
pub fn load_fbx(_path: &Path) -> Result<Scene> {
    bail!("FBX load from path is not available on WebAssembly")
}

/// Parse ASCII FBX text into a BlendCraft scene.
pub fn load_fbx_str(text: &str) -> Result<Scene> {
    let doc = parse_ascii(text)?;
    convert_doc(&doc, &|_| None)
}

/// Load FBX from bytes — auto-detects binary vs ASCII. Only textures embedded in the file
/// are available; use [`load_fbx_bytes_with`] to also resolve external texture files.
pub fn load_fbx_bytes(bytes: &[u8]) -> Result<Scene> {
    load_fbx_bytes_with(bytes, &|_| None)
}

/// Like [`load_fbx_bytes`], with a callback that returns the bytes of an external texture
/// file given the path stored in the FBX (it is called with the relative name, the full
/// name and the bare file name).
pub fn load_fbx_bytes_with(
    bytes: &[u8],
    resolver: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Result<Scene> {
    if is_binary_fbx(bytes) {
        let doc = parse_binary(bytes)?;
        return convert_doc(&doc, resolver);
    }
    let text = std::str::from_utf8(bytes)
        .context("FBX is not valid UTF-8 (and not a binary FBX header)")?;
    let doc = parse_ascii(text)?;
    convert_doc(&doc, resolver)
}

fn is_binary_fbx(bytes: &[u8]) -> bool {
    bytes.len() >= 27 && bytes.starts_with(b"Kaydara FBX Binary")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn export_fbx(scene: &Scene, path: &Path, only: Option<&BTreeSet<u64>>) -> Result<()> {
    // Binary FBX 7.4 — Blender and most DCCs import this; ASCII is often rejected.
    // Write any referenced textures next to the FBX so RelativeFilename links resolve.
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("export");
    let mut tex_rel: HashMap<usize, String> = HashMap::new();
    for (ti, tex) in scene.textures.iter().enumerate() {
        let safe: String = tex
            .name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let safe = if safe.is_empty() {
            format!("tex{ti}")
        } else {
            safe
        };
        let fname = format!("{stem}_{safe}.png");
        let fpath = dir.join(&fname);
        if let Err(e) = image::save_buffer(
            &fpath,
            &tex.rgba8,
            tex.width,
            tex.height,
            image::ColorType::Rgba8,
        ) {
            eprintln!("FBX texture write failed for {}: {e}", fpath.display());
            continue;
        }
        tex_rel.insert(ti, fname);
    }
    let bytes = build_binary_fbx(scene, only, &tex_rel)?;
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Optional ASCII export (debug / tools that prefer text).
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub fn export_fbx_ascii(scene: &Scene, path: &Path, only: Option<&BTreeSet<u64>>) -> Result<()> {
    let text = build_ascii_fbx(scene, only)?;
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

pub fn export_fbx_bytes(scene: &Scene, only: Option<&BTreeSet<u64>>) -> Result<Vec<u8>> {
    build_binary_fbx(scene, only, &HashMap::new())
}

#[cfg(target_arch = "wasm32")]
pub fn export_fbx(scene: &Scene, _path: &Path, only: Option<&BTreeSet<u64>>) -> Result<()> {
    let _ = export_fbx_bytes(scene, only)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Minimal ASCII FBX DOM
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct FbxDoc {
    /// id -> geometry
    geometries: BTreeMap<i64, Geom>,
    /// id -> model
    models: BTreeMap<i64, Model>,
    /// id -> material
    materials: BTreeMap<i64, Mat>,
    /// id -> texture node
    textures: BTreeMap<i64, Tex>,
    /// id -> video (image) node; may carry the embedded image bytes
    videos: BTreeMap<i64, Video>,
    /// id -> Skin deformer
    skins: BTreeMap<i64, SkinDef>,
    /// id -> Cluster sub-deformer
    clusters: BTreeMap<i64, ClusterDef>,
    /// Animation stacks / layers / curve nodes / curves
    anim_stacks: BTreeMap<i64, AnimStackDef>,
    anim_layers: BTreeMap<i64, AnimLayerDef>,
    anim_curve_nodes: BTreeMap<i64, AnimCurveNodeDef>,
    anim_curves: BTreeMap<i64, AnimCurveDef>,
    /// all connections (OO object links and OP property links)
    connections: Vec<Conn>,
    /// GlobalSettings UpAxis (0 = X, 1 = Y, 2 = Z) and its sign
    up_axis: i32,
    up_sign: i32,
    /// Multiply positions/translations by this to get meters.
    /// Derived from GlobalSettings UnitScaleFactor (FBX stores a factor to centimeters).
    unit_to_meters: f32,
}

impl Default for FbxDoc {
    fn default() -> Self {
        Self {
            geometries: BTreeMap::new(),
            models: BTreeMap::new(),
            materials: BTreeMap::new(),
            textures: BTreeMap::new(),
            videos: BTreeMap::new(),
            skins: BTreeMap::new(),
            clusters: BTreeMap::new(),
            anim_stacks: BTreeMap::new(),
            anim_layers: BTreeMap::new(),
            anim_curve_nodes: BTreeMap::new(),
            anim_curves: BTreeMap::new(),
            connections: Vec::new(),
            up_axis: 1,
            up_sign: 1,
            // Default FBX unit is centimeters → meters
            unit_to_meters: 0.01,
        }
    }
}

#[derive(Debug, Default)]
struct Geom {
    name: String,
    vertices: Vec<[f32; 3]>,
    /// Polygon vertex indices (FBX style: last of each poly is bitwise-not)
    polygon_vertex_index: Vec<i32>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    uv_indices: Vec<i32>,
    normal_indices: Vec<i32>,
    /// LayerElement MappingInformationType / ReferenceInformationType
    normal_mapping: String,
    normal_ref: String,
    uv_mapping: String,
    uv_ref: String,
    /// Per-polygon material slot (LayerElementMaterial)
    material_ids: Vec<i32>,
    material_mapping: String,
}

#[derive(Debug, Clone)]
struct Conn {
    /// "OO" (object-object) or "OP" (object-property)
    ty: String,
    child: i64,
    parent: i64,
    /// property name for OP links, e.g. "DiffuseColor"
    prop: String,
}

#[derive(Debug, Default)]
struct Tex {
    name: String,
    file: String,
    relative: String,
}

#[derive(Debug, Default)]
struct Video {
    file: String,
    relative: String,
    /// Embedded image file (png/jpg/...), empty when the texture is external.
    content: Vec<u8>,
}

/// FBX Deformer of type Skin — groups Cluster sub-deformers for one geometry.
#[derive(Debug, Default, Clone)]
struct SkinDef {
    name: String,
}

/// FBX SubDeformer Cluster — per-bone vertex indices + weights and bind matrices.
#[derive(Debug, Default, Clone)]
struct ClusterDef {
    name: String,
    indexes: Vec<i32>,
    weights: Vec<f32>,
    /// 16 floats column-major if present (geometry → bone at bind).
    transform: Option<[f32; 16]>,
    /// 16 floats column-major if present (bone world at bind = TransformLink).
    transform_link: Option<[f32; 16]>,
}

/// FBX time unit: 1 second = 46_186_158_000 KTime ticks (SDK default).
const FBX_TIME_SECOND: f64 = 46_186_158_000.0;

#[derive(Debug, Default, Clone)]
struct AnimStackDef {
    name: String,
    /// Local start/stop in FBX KTime (optional; may be missing).
    local_start: Option<i64>,
    local_stop: Option<i64>,
}

#[derive(Debug, Default, Clone)]
struct AnimLayerDef {
    name: String,
}

/// AnimationCurveNode — usually Lcl Translation / Rotation / Scaling with 3 channels.
#[derive(Debug, Default, Clone)]
struct AnimCurveNodeDef {
    name: String,
    /// Default values for X,Y,Z when no curve is connected (from Properties70).
    defaults: [f32; 3],
}

#[derive(Debug, Default, Clone)]
struct AnimCurveDef {
    name: String,
    /// Parallel key times (seconds) and values.
    times: Vec<f64>,
    values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct Model {
    name: String,
    kind: String, // "Mesh", "Null", ...
    translation: [f32; 3],
    rotation_deg: [f32; 3], // Euler XYZ degrees
    scale: [f32; 3],
    /// Geometric (pivot) transform — applied to geometry only, not inherited by children
    geom_translation: [f32; 3],
    geom_rotation_deg: [f32; 3],
    geom_scale: [f32; 3],
    /// PreRotation / PostRotation (degrees) — bones and Blender/Maya exports use these.
    pre_rotation_deg: [f32; 3],
    post_rotation_deg: [f32; 3],
    /// FBX RotationOrder enum: 0 XYZ, 1 XZY, 2 YZX, 3 YXZ, 4 ZXY, 5 ZYX
    rotation_order: i32,
    parent: Option<i64>,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            name: String::new(),
            kind: String::new(),
            translation: [0.0; 3],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
            geom_translation: [0.0; 3],
            geom_rotation_deg: [0.0; 3],
            geom_scale: [1.0; 3],
            pre_rotation_deg: [0.0; 3],
            post_rotation_deg: [0.0; 3],
            rotation_order: 0,
            parent: None,
        }
    }
}

#[derive(Debug)]
struct Mat {
    name: String,
    diffuse: [f32; 3],
    emissive: [f32; 3],
    opacity: f32,
}

impl Mat {
    fn new(name: String) -> Self {
        Self {
            name,
            diffuse: [0.8, 0.8, 0.8],
            emissive: [0.0; 3],
            opacity: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

fn parse_ascii(text: &str) -> Result<FbxDoc> {
    let mut doc = FbxDoc::default();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0usize;

    while i < lines.len() {
        let line = lines[i].trim();
        if line.starts_with("Geometry:") {
            let (id, name, body, next) = read_block(&lines, i)?;
            i = next;
            let mut g = Geom {
                name: strip_name(&name),
                ..Default::default()
            };
            parse_geometry_body(&body, &mut g);
            doc.geometries.insert(id, g);
            continue;
        }
        if line.starts_with("Model:") {
            let (id, name, body, next) = read_block(&lines, i)?;
            i = next;
            let kind = parse_header_class(line);
            let mut m = Model {
                name: strip_name(&name),
                kind,
                scale: [1.0, 1.0, 1.0],
                ..Default::default()
            };
            parse_model_body(&body, &mut m);
            doc.models.insert(id, m);
            continue;
        }
        if line.starts_with("Material:") {
            let (id, name, body, next) = read_block(&lines, i)?;
            i = next;
            let mut m = Mat::new(strip_name(&name));
            parse_material_body(&body, &mut m);
            doc.materials.insert(id, m);
            continue;
        }
        if line.starts_with("Texture:") {
            let (id, name, body, next) = read_block(&lines, i)?;
            i = next;
            doc.textures.insert(
                id,
                Tex {
                    name: strip_name(&name),
                    file: first_quoted(&body, &["FileName:", "Filename:"]),
                    relative: first_quoted(&body, &["RelativeFilename:"]),
                },
            );
            continue;
        }
        if line.starts_with("Video:") {
            let (id, _name, body, next) = read_block(&lines, i)?;
            i = next;
            doc.videos.insert(
                id,
                Video {
                    file: first_quoted(&body, &["Filename:", "FileName:"]),
                    relative: first_quoted(&body, &["RelativeFilename:"]),
                    content: ascii_embedded_content(&body),
                },
            );
            continue;
        }
        if line.starts_with("Connections:") {
            let (_, _, body, next) = read_block(&lines, i)?;
            i = next;
            for bl in body.lines() {
                let bl = bl.trim();
                // C: "OO",child,parent
                if bl.starts_with("C:") {
                    if let Some(c) = parse_connection(bl) {
                        doc.connections.push(c);
                    }
                }
            }
            continue;
        }
        if line.starts_with("GlobalSettings:") {
            let (_, _, body, next) = read_block(&lines, i)?;
            i = next;
            parse_global_settings_ascii(&body, &mut doc);
            continue;
        }
        i += 1;
    }

    doc.geometries
        .retain(|_, g| !g.polygon_vertex_index.is_empty() && !g.vertices.is_empty());
    if doc.geometries.is_empty() {
        bail!("No Geometry blocks found — is this an ASCII FBX with mesh data?");
    }
    Ok(doc)
}

/// Read a top-level `Type: id, "name", "class" { ... }` block.
fn read_block<'a>(lines: &[&'a str], start: usize) -> Result<(i64, String, String, usize)> {
    let header = lines[start].trim();
    let id = parse_header_id(header).unwrap_or(0);
    let name = parse_header_name(header).unwrap_or_default();

    // Find opening brace on this or next lines
    let mut i = start;
    let mut depth = 0i32;
    let mut started = false;
    let mut body = String::new();
    while i < lines.len() {
        let l = lines[i];
        for ch in l.chars() {
            if ch == '{' {
                depth += 1;
                started = true;
                if depth == 1 {
                    continue; // don't include the outer opening brace content marker
                }
            }
            if started && depth >= 1 {
                if ch == '{' {
                    body.push(ch);
                } else if ch == '}' {
                    depth -= 1;
                    if depth == 0 {
                        return Ok((id, name, body, i + 1));
                    }
                    body.push(ch);
                } else if depth >= 1 {
                    body.push(ch);
                }
            } else if ch == '}' {
                depth -= 1;
            }
        }
        if started && depth >= 1 {
            body.push('\n');
        }
        i += 1;
    }
    bail!("Unclosed FBX block starting at line {}", start + 1)
}

fn parse_header_id(header: &str) -> Option<i64> {
    // Geometry: 123, "Geometry::Cube", "Mesh" {
    let after = header.split(':').nth(1)?;
    let num = after.split(',').next()?.trim();
    num.parse().ok()
}

fn parse_header_name(header: &str) -> Option<String> {
    let start = header.find('"')?;
    let end = header[start + 1..].find('"')? + start + 1;
    Some(header[start + 1..end].to_string())
}

fn strip_name(full: &str) -> String {
    // "Geometry::Cube" or "Model::Cube" -> Cube
    full.rsplit("::").next().unwrap_or(full).to_string()
}

/// Second quoted field (the class) of a block header: `Model: 1, "Model::Arm", "LimbNode" {` -> LimbNode.
fn parse_header_class(header: &str) -> String {
    header
        .split('"')
        .skip(1)
        .step_by(2)
        .nth(1)
        .unwrap_or("")
        .to_string()
}

fn parse_connection(line: &str) -> Option<Conn> {
    // C: "OO",123,456      C: "OP",123,456, "DiffuseColor"
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 3 {
        return None;
    }
    let ty = parts[0]
        .trim_start_matches("C:")
        .trim()
        .trim_matches('"')
        .to_string();
    let child = parts[1].trim().parse().ok()?;
    let parent = parts[2].trim().split_whitespace().next()?.parse().ok()?;
    let prop = parts
        .get(3)
        .map(|p| p.trim().trim_matches('"').to_string())
        .unwrap_or_default();
    Some(Conn {
        ty,
        child,
        parent,
        prop,
    })
}

/// First quoted string after any of the labels (empty if none).
fn first_quoted(body: &str, labels: &[&str]) -> String {
    for l in labels {
        let Some(i) = body.find(l) else { continue };
        let rest = body[i + l.len()..].trim_start();
        if let Some(r) = rest.strip_prefix('"') {
            if let Some(e) = r.find('"') {
                return r[..e].to_string();
            }
        }
    }
    String::new()
}

/// Base64 `Content: "...."` of an ASCII Video node.
fn ascii_embedded_content(body: &str) -> Vec<u8> {
    let Some(i) = body.find("Content:") else {
        return Vec::new();
    };
    let rest = body[i + 8..].trim_start();
    let Some(r) = rest.strip_prefix('"') else {
        return Vec::new();
    };
    let Some(e) = r.find('"') else {
        return Vec::new();
    };
    base64_decode(&r[..e])
}

fn base64_decode(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => continue, // padding / whitespace
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    out
}

/// The text of one `LayerElementXxx:` block (up to the next layer element).
fn layer_section<'a>(body: &'a str, label: &str) -> Option<&'a str> {
    let i = body.find(label)?;
    let rest = &body[i + label.len()..];
    let mut end = rest.len();
    for k in ["LayerElement", "Layer:"] {
        if let Some(j) = rest.find(k) {
            end = end.min(j);
        }
    }
    Some(&rest[..end])
}

fn parse_geometry_body(body: &str, g: &mut Geom) {
    if let Some(v) = extract_array_f32(body, "Vertices:") {
        for chunk in v.chunks_exact(3) {
            g.vertices.push([chunk[0], chunk[1], chunk[2]]);
        }
    }
    if let Some(v) = extract_array_i32(body, "PolygonVertexIndex:") {
        g.polygon_vertex_index = v;
    }
    if let Some(sec) = layer_section(body, "LayerElementNormal:") {
        if let Some(v) = extract_array_f32(sec, "Normals:") {
            for chunk in v.chunks_exact(3) {
                g.normals.push([chunk[0], chunk[1], chunk[2]]);
            }
        }
        if let Some(v) = extract_array_i32(sec, "NormalsIndex:") {
            g.normal_indices = v;
        }
        g.normal_mapping = first_quoted(sec, &["MappingInformationType:"]);
        g.normal_ref = first_quoted(sec, &["ReferenceInformationType:"]);
    }
    if let Some(sec) = layer_section(body, "LayerElementUV:") {
        if let Some(v) = extract_array_f32(sec, "UV:") {
            for chunk in v.chunks_exact(2) {
                g.uvs.push([chunk[0], chunk[1]]);
            }
        }
        if let Some(v) = extract_array_i32(sec, "UVIndex:") {
            g.uv_indices = v;
        }
        g.uv_mapping = first_quoted(sec, &["MappingInformationType:"]);
        g.uv_ref = first_quoted(sec, &["ReferenceInformationType:"]);
    }
    if let Some(sec) = layer_section(body, "LayerElementMaterial:") {
        if let Some(v) = extract_array_i32(sec, "Materials:") {
            g.material_ids = v;
        }
        g.material_mapping = first_quoted(sec, &["MappingInformationType:"]);
    }
}

fn parse_model_body(body: &str, m: &mut Model) {
    for key in [
        "Lcl Translation",
        "Lcl Rotation",
        "Lcl Scaling",
        "GeometricTranslation",
        "GeometricRotation",
        "GeometricScaling",
    ] {
        if let Some(vals) = extract_property_vec3(body, key) {
            match key {
                "Lcl Translation" => m.translation = vals,
                "Lcl Rotation" => m.rotation_deg = vals,
                "Lcl Scaling" => m.scale = vals,
                "GeometricTranslation" => m.geom_translation = vals,
                "GeometricRotation" => m.geom_rotation_deg = vals,
                "GeometricScaling" => m.geom_scale = vals,
                _ => {}
            }
        }
    }
    if let Some(v) = extract_property_vec3(body, "PreRotation") {
        m.pre_rotation_deg = v;
    }
    if let Some(v) = extract_property_vec3(body, "PostRotation") {
        m.post_rotation_deg = v;
    }
    if let Some(v) = extract_property_scalar(body, "RotationOrder") {
        m.rotation_order = v as i32;
    }
}

fn parse_material_body(body: &str, m: &mut Mat) {
    if let Some(v) = extract_property_vec3(body, "DiffuseColor") {
        m.diffuse = v;
    } else if let Some(v) = extract_property_vec3(body, "Diffuse") {
        m.diffuse = v;
    }
    if let Some(v) = extract_property_vec3(body, "EmissiveColor") {
        m.emissive = v;
    }
    if let Some(v) = extract_property_scalar(body, "Opacity") {
        m.opacity = v.clamp(0.0, 1.0);
    }
}

fn extract_property_vec3(body: &str, key: &str) -> Option<[f32; 3]> {
    for line in body.lines() {
        let t = line.trim();
        if t.contains(key) && t.starts_with("P:") {
            // numbers are the last 3 comma-separated floats
            let nums: Vec<f32> = t
                .split(',')
                .filter_map(|s| {
                    let s = s.trim().trim_matches('"');
                    s.parse::<f32>().ok()
                })
                .collect();
            if nums.len() >= 3 {
                let n = nums.len();
                return Some([nums[n - 3], nums[n - 2], nums[n - 1]]);
            }
        }
    }
    None
}

fn extract_array_f32(body: &str, label: &str) -> Option<Vec<f32>> {
    extract_array_raw(body, label)?
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect::<Vec<_>>()
        .into()
}

fn extract_array_i32(body: &str, label: &str) -> Option<Vec<i32>> {
    extract_array_raw(body, label)?
        .split(',')
        .filter_map(|s| s.trim().parse::<i32>().ok())
        .collect::<Vec<_>>()
        .into()
}

/// Find `Label: *N { a: 1,2,3 }` or multi-line and return the number list string.
fn extract_array_raw(body: &str, label: &str) -> Option<String> {
    let idx = body.find(label)?;
    let after = &body[idx + label.len()..];
    // Find "a:"
    let a_idx = after.find("a:")?;
    let mut rest = &after[a_idx + 2..];
    // Collect until we hit a line that is only "}" or a new property key without commas of numbers
    let mut buf = String::new();
    for line in rest.lines() {
        let t = line.trim();
        if t == "}" || t.ends_with('{') {
            // end of array block
            // still take numbers on same line before }
            let cleaned = t.trim_end_matches('}').trim();
            if !cleaned.is_empty() && cleaned != "{" {
                if !buf.is_empty() {
                    buf.push(',');
                }
                buf.push_str(cleaned);
            }
            break;
        }
        // stop if we see another property like "PolygonVertexIndex:" on its own header
        if t.contains(':')
            && !t.contains(',')
            && !t
                .chars()
                .next()
                .map(|c| c.is_ascii_digit() || c == '-' || c == '.')
                .unwrap_or(false)
        {
            // might be next property — if we already have data, stop
            if !buf.is_empty() && !t.starts_with('a') {
                break;
            }
        }
        if !buf.is_empty() {
            buf.push(',');
        }
        buf.push_str(t.trim_end_matches(','));
        // safety limit
        if buf.len() > 50_000_000 {
            break;
        }
    }
    // Also handle single-line: a: 1,2,3 }
    if buf.is_empty() {
        rest = rest.trim_start();
        let end = rest.find('}').unwrap_or(rest.len().min(10_000));
        buf = rest[..end].replace('\n', ",").trim().to_string();
    }
    if buf.is_empty() {
        None
    } else {
        Some(buf)
    }
}

// ---------------------------------------------------------------------------
// Convert DOM → BlendCraft Scene
// ---------------------------------------------------------------------------

/// FBX UnitScaleFactor = how many centimeters one file unit is.
/// Convert to meters: unit_to_meters = UnitScaleFactor * 0.01
fn unit_factor_to_meters(unit_scale_factor: f32) -> f32 {
    let cm = unit_scale_factor;
    // Clamp nonsense values
    if !cm.is_finite() || cm.abs() < 1e-12 {
        return 0.01;
    }
    cm * 0.01
}

fn parse_global_settings_ascii(body: &str, doc: &mut FbxDoc) {
    // P: "UnitScaleFactor", "double", "Number", "",1  or similar
    if let Some(v) = extract_property_scalar(body, "UnitScaleFactor") {
        doc.unit_to_meters = unit_factor_to_meters(v);
    }
    if let Some(v) = extract_property_scalar(body, "UpAxis") {
        doc.up_axis = v as i32;
    }
    if let Some(v) = extract_property_scalar(body, "UpAxisSign") {
        doc.up_sign = if v < 0.0 { -1 } else { 1 };
    }
}

fn extract_property_scalar(body: &str, key: &str) -> Option<f32> {
    let quoted = format!("\"{key}\"");
    for line in body.lines() {
        let t = line.trim();
        if t.starts_with("P:") && t.contains(&quoted) {
            let nums: Vec<f32> = t
                .split(',')
                .filter_map(|s| {
                    let s = s.trim().trim_matches('"');
                    s.parse::<f32>().ok()
                })
                .collect();
            if let Some(&last) = nums.last() {
                return Some(last);
            }
        }
    }
    None
}

/// Rotation from Euler degrees in an FBX rotation order. FBX names the order in which the
/// axes are applied, so "XYZ" is the matrix Rz * Ry * Rx.
fn euler_quat(deg: [f32; 3], order: i32) -> Quat {
    let qx = Quat::from_rotation_x(deg[0].to_radians());
    let qy = Quat::from_rotation_y(deg[1].to_radians());
    let qz = Quat::from_rotation_z(deg[2].to_radians());
    match order {
        1 => qy * qz * qx, // XZY
        2 => qx * qz * qy, // YZX
        3 => qz * qx * qy, // YXZ
        4 => qy * qx * qz, // ZXY
        5 => qx * qy * qz, // ZYX
        _ => qz * qy * qx, // XYZ (default)
    }
}

fn model_matrix(t: [f32; 3], rot_deg: [f32; 3], s: [f32; 3]) -> Mat4 {
    let q = euler_quat(rot_deg, 0);
    Mat4::from_scale_rotation_translation(Vec3::from(s), q, Vec3::from(t))
}

/// Node-to-parent matrix of a Model: T * Rpre * R * Rpost^-1 * S (pivots/offsets ignored).
fn local_matrix(m: &Model) -> Mat4 {
    let rot = euler_quat(m.pre_rotation_deg, 0)
        * euler_quat(m.rotation_deg, m.rotation_order)
        * euler_quat(m.post_rotation_deg, 0).inverse();
    Mat4::from_scale_rotation_translation(
        Vec3::from(m.scale),
        rot,
        Vec3::from(m.translation),
    )
}

/// World matrix of a Model in file units: walks up the Model -> Model parent links, so a mesh
/// parented to a bone / null / armature ends up where it belongs.
fn model_world(id: i64, doc: &FbxDoc, parents: &HashMap<i64, i64>) -> Mat4 {
    let mut chain: Vec<&Model> = Vec::new();
    let mut seen: BTreeSet<i64> = BTreeSet::new();
    let mut cur = Some(id);
    while let Some(c) = cur {
        if !seen.insert(c) {
            break; // cycle guard
        }
        let Some(m) = doc.models.get(&c) else { break };
        chain.push(m);
        cur = parents.get(&c).copied();
    }
    let mut w = Mat4::IDENTITY;
    for m in chain.iter().rev() {
        w = w * local_matrix(m);
    }
    w
}

/// Candidate paths for a texture: relative name, stored full name, then bare file name.
fn texture_candidates(a: &str, b: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for n in [a, b] {
        let n = n.trim();
        if n.is_empty() {
            continue;
        }
        out.push(n.to_string());
        let norm = n.replace('\\', "/");
        if let Some(f) = norm.rsplit('/').next() {
            if !f.is_empty() {
                out.push(f.to_string());
            }
        }
    }
    out.dedup();
    out
}

/// Load (once) the image behind an FBX Texture node into `textures` and return its index.
fn texture_index(
    doc: &FbxDoc,
    tid: i64,
    resolver: TexResolver,
    textures: &mut Vec<TextureData>,
    cache: &mut HashMap<i64, Option<usize>>,
) -> Option<usize> {
    if let Some(c) = cache.get(&tid) {
        return *c;
    }
    let result = (|| {
        let t = doc.textures.get(&tid)?;
        // Video node linked to the texture: embedded bytes and/or additional file names.
        let video = doc
            .connections
            .iter()
            .filter(|c| c.parent == tid)
            .find_map(|c| doc.videos.get(&c.child));
        let mut hint = if !t.file.is_empty() {
            t.file.clone()
        } else {
            t.relative.clone()
        };
        if hint.is_empty() {
            if let Some(v) = video {
                hint = if !v.file.is_empty() { v.file.clone() } else { v.relative.clone() };
            }
        }
        if hint.is_empty() {
            hint = t.name.clone();
        }
        let mut bytes: Option<Vec<u8>> = None;
        if let Some(v) = video {
            if !v.content.is_empty() {
                bytes = Some(v.content.clone());
            }
        }
        if bytes.is_none() {
            let mut names = texture_candidates(&t.relative, &t.file);
            if let Some(v) = video {
                names.extend(texture_candidates(&v.relative, &v.file));
            }
            for n in names {
                if let Some(b) = resolver(&n) {
                    bytes = Some(b);
                    break;
                }
            }
        }
        let Some(bytes) = bytes else {
            eprintln!("FBX texture '{hint}' was not found next to the model");
            return None;
        };
        let mut tex = crate::decode_image_bytes(&bytes, &hint)?;
        if tex.name.is_empty() {
            tex.name = t.name.clone();
        }
        textures.push(tex);
        Some(textures.len() - 1)
    })();
    cache.insert(tid, result);
    result
}

fn convert_doc(doc: &FbxDoc, resolver: TexResolver) -> Result<Scene> {
    let mut scene = Scene::default();
    let u = doc.unit_to_meters;
    // Guard against absurd scales from bad UnitScaleFactor
    let u = if u.is_finite() && u > 1e-9 && u < 1e6 {
        u
    } else {
        0.01
    };

    // Z-up (3ds Max, some Maya/CAD exports) -> the app's Y-up.
    let root_fix = match (doc.up_axis, doc.up_sign >= 0) {
        (2, true) => Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2),
        (2, false) => Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2),
        (0, true) => Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2),
        (0, false) => Mat4::from_rotation_z(-std::f32::consts::FRAC_PI_2),
        _ => Mat4::IDENTITY,
    };

    // ---- connections ----------------------------------------------------------------
    let mut geom_to_models: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut model_mats: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut parents: HashMap<i64, i64> = HashMap::new();
    for c in &doc.connections {
        let (child, parent) = (c.child, c.parent);
        if doc.geometries.contains_key(&child) && doc.models.contains_key(&parent) {
            geom_to_models.entry(child).or_default().push(parent);
        }
        if doc.materials.contains_key(&child) && doc.models.contains_key(&parent) {
            let v = model_mats.entry(parent).or_default();
            if !v.contains(&child) {
                v.push(child);
            }
        }
        // sometimes material connected the other way
        if doc.materials.contains_key(&parent) && doc.models.contains_key(&child) {
            let v = model_mats.entry(child).or_default();
            if !v.contains(&parent) {
                v.push(parent);
            }
        }
        // Model -> Model: scene hierarchy (bones, nulls, armatures, groups)
        if c.ty != "OP" && doc.models.contains_key(&child) && doc.models.contains_key(&parent) {
            parents.insert(child, parent);
        }
    }

    // ---- materials + textures ------------------------------------------------------
    let mut textures: Vec<TextureData> = Vec::new();
    let mut tex_cache: HashMap<i64, Option<usize>> = HashMap::new();
    let mut mat_id_to_index: HashMap<i64, usize> = HashMap::new();
    for (&id, m) in &doc.materials {
        let idx = scene.materials.len();
        mat_id_to_index.insert(id, idx);
        let mut mat = Material {
            name: m.name.clone(),
            base_color: [m.diffuse[0], m.diffuse[1], m.diffuse[2], m.opacity],
            emissive: m.emissive,
            roughness: 0.5,
            ..Default::default()
        };
        if m.opacity < 0.999 {
            mat.alpha_mode = forge_core::AlphaMode::Blend;
        }
        // Textures hang off the material through OP links named after the slot.
        for c in doc.connections.iter().filter(|c| {
            c.parent == id && doc.textures.contains_key(&c.child)
        }) {
            let slot = c.prop.to_ascii_lowercase();
            let is_base = slot.contains("diffuse")
                || slot.contains("basecolor")
                || slot.contains("base_color")
                || slot.contains("color_map")
                || slot.is_empty();
            let is_normal = slot.contains("normal");
            let is_emissive = slot.contains("emissive") || slot.contains("emission");
            let is_occlusion = slot.contains("occlusion") || slot.contains("ao_map");
            if !(is_base || is_normal || is_emissive || is_occlusion) {
                continue; // bump / specular / reflection maps have no equivalent slot
            }
            let Some(ti) = texture_index(doc, c.child, resolver, &mut textures, &mut tex_cache)
            else {
                continue;
            };
            if is_normal {
                mat.normal_tex = Some(ti);
            } else if is_emissive {
                mat.emissive_tex = Some(ti);
                mat.emissive = [1.0, 1.0, 1.0];
            } else if is_occlusion {
                mat.occlusion_tex = Some(ti);
            } else if mat.base_color_tex.is_none() {
                mat.base_color_tex = Some(ti);
                // The diffuse colour multiplies the texture; exporters leave a grey default
                // there, which would darken the image.
                mat.base_color = [1.0, 1.0, 1.0, m.opacity];
            }
        }
        scene.materials.push(mat);
    }
    if scene.materials.is_empty() {
        scene.materials.push(Material::default());
    }

    // ---- objects -------------------------------------------------------------------
    // object name → control-point index per mesh vertex (for cluster Indexes)
    let mut mesh_ctrl_points: HashMap<String, Vec<usize>> = HashMap::new();
    for (&gid, geom) in &doc.geometries {
        let (base_mesh, tri_slot, ctrl_of_vertex) = geom_to_mesh(geom)?;
        if base_mesh.indices.is_empty() || base_mesh.vertices.is_empty() {
            continue;
        }
        // Instances: the same geometry can be shared by several models.
        let model_ids: Vec<Option<i64>> = match geom_to_models.get(&gid) {
            Some(v) if !v.is_empty() => v.iter().map(|&m| Some(m)).collect(),
            _ => vec![None],
        };
        for mid in model_ids {
            let model = mid
                .and_then(|m| doc.models.get(&m))
                .cloned()
                .unwrap_or_else(|| Model {
                    name: geom.name.clone(),
                    kind: "Mesh".into(),
                    ..Default::default()
                });
            let mut mesh = base_mesh.clone();
            // Geometric (pivot) transform: applied to the geometry only, in file units.
            let geom_tf = model_matrix(
                model.geom_translation,
                model.geom_rotation_deg,
                model.geom_scale,
            );
            if geom_tf != Mat4::IDENTITY {
                mesh = mesh.transformed(geom_tf);
            }
            // Unit conversion: scale vertex positions to meters
            if (u - 1.0).abs() > 1e-8 {
                for v in &mut mesh.vertices {
                    v.position[0] *= u;
                    v.position[1] *= u;
                    v.position[2] *= u;
                }
            }

            // Material slots of this model -> scene material indices
            let slots: Vec<Option<usize>> = mid
                .and_then(|m| model_mats.get(&m))
                .map(|v| v.iter().map(|id| mat_id_to_index.get(id).copied()).collect())
                .unwrap_or_default();
            let slot_material = |s: u32| -> Option<usize> {
                slots.get(s as usize).copied().flatten().or_else(|| {
                    // fewer slots than the mesh refers to: fall back to the first one
                    slots.first().copied().flatten()
                })
            };
            let used: BTreeSet<u32> = tri_slot.iter().copied().collect();
            let material: Option<usize>;
            if used.len() <= 1 {
                material = slot_material(used.iter().next().copied().unwrap_or(0));
            } else {
                // Several materials on one mesh: sort triangles by slot and emit groups.
                let ntri = mesh.indices.len() / 3;
                let mut order: Vec<usize> = (0..ntri).collect();
                order.sort_by_key(|&t| tri_slot.get(t).copied().unwrap_or(0));
                let mut indices = Vec::with_capacity(mesh.indices.len());
                let mut groups: Vec<MeshGroup> = Vec::new();
                let mut run_slot: Option<u32> = None;
                for &t in &order {
                    let sl = tri_slot.get(t).copied().unwrap_or(0);
                    if run_slot != Some(sl) {
                        groups.push(MeshGroup {
                            first_index: indices.len() as u32,
                            index_count: 0,
                            material: slot_material(sl),
                        });
                        run_slot = Some(sl);
                    }
                    indices.extend_from_slice(&mesh.indices[t * 3..t * 3 + 3]);
                    if let Some(g) = groups.last_mut() {
                        g.index_count += 3;
                    }
                }
                mesh.indices = indices;
                material = groups.first().and_then(|g| g.material);
                mesh.groups = groups;
            }

            // World transform; unit scale applied uniformly (glTF-compatible scene units).
            let world = match mid {
                Some(m) => root_fix * model_world(m, doc, &parents),
                None => root_fix,
            };
            let world = if (u - 1.0).abs() > 1e-8 {
                Mat4::from_scale(Vec3::splat(u)) * world
            } else {
                world
            };
            let (scale, rotation, translation) = world.to_scale_rotation_translation();
            let mut obj = Object::new(model.name.clone(), mesh);
            obj.translation = translation;
            obj.rotation = rotation;
            obj.scale = scale;
            obj.material = material;
            mesh_ctrl_points.insert(model.name.to_ascii_lowercase(), ctrl_of_vertex.clone());
            scene.objects.push(obj);
        }
    }

    // Reconstruct FBX LimbNode / Root / Null models as editable armature bones.
    // FBX stores bones as Model objects connected by OO hierarchy links.
    // Accept the class names used by Blender, Maya, 3ds Max and Unity. Also
    // pull in Null/Locator nodes when the file already contains at least one
    // explicit LimbNode so pure-hierarchy skeletons still produce bones.
    let has_explicit_limb = doc.models.values().any(|m| {
        let k = m.kind.to_ascii_lowercase();
        k == "limbnode" || k == "limb" || k == "root"
    });
    let limb_ids: Vec<i64> = doc.models.iter()
        .filter_map(|(&id, m)| {
            let k = m.kind.to_ascii_lowercase();
            let is_bone_class = k == "limbnode" || k == "root" || k == "limb"
                || k == "bone" || k.contains("limb");
            let is_null_in_skeleton = has_explicit_limb
                && (k == "null" || k == "locator" || k == "empty");
            (is_bone_class || is_null_in_skeleton).then_some(id)
        })
        .collect();
    let mut _armature_object_id: Option<u64> = None;
    if !limb_ids.is_empty() {
        let id_to_bone: HashMap<i64, u32> = limb_ids.iter().enumerate().map(|(i, id)| (*id, i as u32)).collect();
        let mut arm = forge_core::ArmatureData { bones: Vec::new(), selected: BTreeSet::new(), active: None };
        let mut heads = HashMap::new();
        for &mid in &limb_ids {
            let w = root_fix * model_world(mid, doc, &parents);
            heads.insert(mid, w.transform_point3(Vec3::ZERO) * u);
        }
        for &mid in &limb_ids {
            let head = heads.get(&mid).copied().unwrap_or(Vec3::ZERO);
            let child_head = limb_ids.iter().find(|&&cid| parents.get(&cid) == Some(&mid)).and_then(|cid| heads.get(cid)).copied();
            let tail = child_head.filter(|p| p.distance(head) > 1e-5).unwrap_or(head + Vec3::Y * 0.1);
            let model = &doc.models[&mid];
            let mut bone = forge_core::Bone::new(model.name.clone(), head, tail);
            bone.parent = parents.get(&mid).and_then(|p| id_to_bone.get(p)).copied();
            // Delta pose mode: pose_* is offset from rest (head/tail). Avoids absolute
            // local TRS fighting hierarchical rest and stretching the skin at frame 0.
            let rot = euler_deg_to_quat(model.rotation_deg, model.rotation_order);
            bone.rest_orientation = Some(rot.to_array());
            arm.bones.push(bone);
        }
        if !arm.bones.is_empty() {
            arm.active = Some(0);
            arm.selected.insert(0);
            let mut arm_obj = Object::new("Armature", MeshData::default());
            arm_obj.id = scene.alloc_id();
            arm_obj.kind = ObjectKind::Armature(arm);
            _armature_object_id = Some(arm_obj.id);
            scene.objects.push(arm_obj);
        }
    }

    // Pure-Rust skin weights (replaces former ufbx SkinDeformer path).
    if let Some(arm_id) = _armature_object_id {
        apply_fbx_skins(&mut scene, doc, &parents, arm_id, &mesh_ctrl_points);
        sample_fbx_animation(&mut scene, doc, &parents, arm_id);
    }

    if scene.objects.is_empty() {
        bail!("FBX contained no usable mesh or armature objects");
    }
    scene.textures = Arc::new(textures);
    // normalize_fbx_scene_scale disabled: auto 0.01 shrink desynced mesh vs armature.
    Ok(scene)
}

/// If the imported FBX is clearly in centimetres (common when UnitScaleFactor is wrong)
/// shrink it so it matches typical glTF/GLB metre-scale assets.
fn normalize_fbx_scene_scale(scene: &mut Scene) {
    let mut max_e = 0.0f32;
    for o in &scene.objects {
        for v in &o.mesh.vertices {
            max_e = max_e.max(v.position[0].abs()).max(v.position[1].abs()).max(v.position[2].abs());
        }
        max_e = max_e
            .max(o.translation[0].abs())
            .max(o.translation[1].abs())
            .max(o.translation[2].abs());
        if let ObjectKind::Armature(arm) = &o.kind {
            for b in &arm.bones {
                max_e = max_e
                    .max(b.head[0].abs())
                    .max(b.head[1].abs())
                    .max(b.head[2].abs())
                    .max(b.tail[0].abs())
                    .max(b.tail[1].abs())
                    .max(b.tail[2].abs());
            }
        }
    }
    // Characters / props in metres rarely exceed ~20 units; cm models are hundreds+.
    if max_e < 50.0 {
        return;
    }
    let s = 0.01f32;
    let scale = Mat4::from_scale(Vec3::splat(s));
    for o in &mut scene.objects {
        for v in &mut Arc::make_mut(&mut o.mesh).vertices {
            v.position[0] *= s;
            v.position[1] *= s;
            v.position[2] *= s;
        }
        o.translation *= s;
        if let ObjectKind::Armature(arm) = &mut o.kind {
            for b in &mut arm.bones {
                b.head = (Vec3::from(b.head) * s).to_array();
                b.tail = (Vec3::from(b.tail) * s).to_array();
                if let Some(loc) = b.rest_local_loc.as_mut() {
                    loc[0] *= s;
                    loc[1] *= s;
                    loc[2] *= s;
                }
                b.pose_loc[0] *= s;
                b.pose_loc[1] *= s;
                b.pose_loc[2] *= s;
            }
        }
        if let Some(skin) = o.skin.as_mut() {
            for p in &mut skin.rest_positions {
                p[0] *= s;
                p[1] *= s;
                p[2] *= s;
            }
            for m in &mut skin.inverse_binds {
                m[12] *= s;
                m[13] *= s;
                m[14] *= s;
            }
        }
    }
    // Scale animation location keys
    for clip in &mut scene.animation.bone_tracks {
        for bones in clip.values_mut() {
            for track in bones.values_mut() {
                for k in &mut track.keys {
                    if let Some(loc) = k.location.as_mut() {
                        *loc *= s;
                    }
                }
            }
        }
    }
    let _ = scale;
}


/// Sample FBX AnimationStacks into `scene.animation.bone_tracks` (desktop + wasm).
fn sample_fbx_animation(
    scene: &mut Scene,
    doc: &FbxDoc,
    parents: &HashMap<i64, i64>,
    arm_id: u64,
) {
    if doc.anim_stacks.is_empty() || doc.anim_curves.is_empty() {
        return;
    }
    let fps = scene.animation.fps.max(1) as f64;

    // Bone model id → bone index (same classification as armature build)
    let has_explicit_limb = doc.models.values().any(|m| {
        let k = m.kind.to_ascii_lowercase();
        k == "limbnode" || k == "limb" || k == "root"
    });
    let mut id_to_bone: HashMap<i64, u32> = HashMap::new();
    let mut bi = 0u32;
    for (id, m) in &doc.models {
        let k = m.kind.to_ascii_lowercase();
        let is_bone = k == "limbnode"
            || k == "root"
            || k == "limb"
            || k == "bone"
            || k.contains("limb");
        let is_null = has_explicit_limb && (k == "null" || k == "locator" || k.is_empty());
        if is_bone || is_null {
            id_to_bone.insert(*id, bi);
            bi += 1;
        }
    }
    if id_to_bone.is_empty() {
        return;
    }

    // layer → stack (OO both directions)
    let mut layer_to_stack: HashMap<i64, i64> = HashMap::new();
    // curve → curve_node + channel prop (OP)
    let mut curve_to_node: HashMap<i64, (i64, String)> = HashMap::new();
    // curve_node → (model id, property e.g. "Lcl Translation")
    let mut node_to_model: HashMap<i64, (i64, String)> = HashMap::new();
    // curve_node → layer
    let mut node_to_layer: HashMap<i64, i64> = HashMap::new();

    for c in &doc.connections {
        if c.ty == "OO" {
            if doc.anim_layers.contains_key(&c.child) && doc.anim_stacks.contains_key(&c.parent) {
                layer_to_stack.insert(c.child, c.parent);
            }
            if doc.anim_layers.contains_key(&c.parent) && doc.anim_stacks.contains_key(&c.child) {
                layer_to_stack.insert(c.parent, c.child);
            }
            if doc.anim_curve_nodes.contains_key(&c.child) && doc.anim_layers.contains_key(&c.parent) {
                node_to_layer.insert(c.child, c.parent);
            }
            if doc.anim_curve_nodes.contains_key(&c.parent) && doc.anim_layers.contains_key(&c.child) {
                node_to_layer.insert(c.parent, c.child);
            }
        } else if c.ty == "OP" {
            if doc.anim_curves.contains_key(&c.child) && doc.anim_curve_nodes.contains_key(&c.parent) {
                curve_to_node.insert(c.child, (c.parent, c.prop.clone()));
            }
            if doc.anim_curves.contains_key(&c.parent) && doc.anim_curve_nodes.contains_key(&c.child) {
                curve_to_node.insert(c.parent, (c.child, c.prop.clone()));
            }
            if doc.anim_curve_nodes.contains_key(&c.child) && doc.models.contains_key(&c.parent) {
                node_to_model.insert(c.child, (c.parent, c.prop.clone()));
            }
            if doc.anim_curve_nodes.contains_key(&c.parent) && doc.models.contains_key(&c.child) {
                node_to_model.insert(c.parent, (c.child, c.prop.clone()));
            }
        }
    }

    // Build per-stack list of (bone_idx, channel kind, component 0/1/2, curve id)
    // channel kind: 0=loc 1=rot 2=scale
    #[derive(Clone)]
    struct Chan {
        bone: u32,
        kind: u8,
        axis: usize,
        curve: i64,
        default: f32,
    }

    let mut stack_chans: HashMap<i64, Vec<Chan>> = HashMap::new();
    for (curve_id, (node_id, prop)) in &curve_to_node {
        let Some((model_id, model_prop)) = node_to_model.get(node_id) else { continue };
        let Some(&bone) = id_to_bone.get(model_id) else { continue };
        let Some(&layer) = node_to_layer.get(node_id) else { continue };
        let Some(&stack) = layer_to_stack.get(&layer) else {
            // Some files connect curve nodes to stack without a layer; try any stack.
            if let Some((sid, _)) = doc.anim_stacks.iter().next() {
                let axis = channel_axis(prop);
                let kind = channel_kind(model_prop);
                let default = doc
                    .anim_curve_nodes
                    .get(node_id)
                    .map(|n| n.defaults[axis])
                    .unwrap_or(0.0);
                stack_chans.entry(*sid).or_default().push(Chan {
                    bone,
                    kind,
                    axis,
                    curve: *curve_id,
                    default,
                });
            }
            continue;
        };
        let axis = channel_axis(prop);
        let kind = channel_kind(model_prop);
        let default = doc
            .anim_curve_nodes
            .get(node_id)
            .map(|n| n.defaults[axis])
            .unwrap_or(0.0);
        stack_chans.entry(stack).or_default().push(Chan {
            bone,
            kind,
            axis,
            curve: *curve_id,
            default,
        });
    }

    if stack_chans.is_empty() {
        // Fallback: no OP wiring found — nothing to sample.
        return;
    }

    for (stack_i, (stack_id, stack)) in doc.anim_stacks.iter().enumerate() {
        let Some(chans) = stack_chans.get(stack_id) else { continue };
        if chans.is_empty() {
            continue;
        }
        // Determine time range from stack props or from keys
        let mut t0 = stack
            .local_start
            .map(|k| k as f64 / FBX_TIME_SECOND)
            .unwrap_or(f64::INFINITY);
        let mut t1 = stack
            .local_stop
            .map(|k| k as f64 / FBX_TIME_SECOND)
            .unwrap_or(f64::NEG_INFINITY);
        for ch in chans {
            if let Some(curve) = doc.anim_curves.get(&ch.curve) {
                if let (Some(&a), Some(&b)) = (curve.times.first(), curve.times.last()) {
                    t0 = t0.min(a);
                    t1 = t1.max(b);
                }
            }
        }
        if !(t1 > t0) || !t0.is_finite() || !t1.is_finite() {
            continue;
        }
        let n_frames = (((t1 - t0) * fps).ceil() as i32).max(1);
        // Cap very long clips so browser import stays responsive
        let n_frames = n_frames.min(10_000);
        let frame_start = 1i32;
        let frame_end = frame_start + n_frames;
        let clip_name = if stack.name.is_empty() {
            format!("AnimStack {}", stack_i + 1)
        } else {
            stack.name.clone()
        };
        scene.animation.clips.push(AnimationClip {
            name: clip_name,
            frame_start,
            frame_end,
        });

        let mut clip_tracks: HashMap<u64, HashMap<u32, ObjectAnimation>> = HashMap::new();
        // Per-bone accumulated LRS for each frame
        for fi in 0..=n_frames {
            let time = t0 + (fi as f64) / fps;
            let frame = frame_start + fi;
            // bone → (loc, euler_deg, scale) components present
            let mut loc: HashMap<u32, [Option<f32>; 3]> = HashMap::new();
            let mut rot: HashMap<u32, [Option<f32>; 3]> = HashMap::new();
            let mut scl: HashMap<u32, [Option<f32>; 3]> = HashMap::new();
            for ch in chans {
                let v = doc
                    .anim_curves
                    .get(&ch.curve)
                    .map(|c| eval_anim_curve(c, time))
                    .unwrap_or(ch.default);
                let slot = match ch.kind {
                    0 => loc.entry(ch.bone).or_insert([None; 3]),
                    1 => rot.entry(ch.bone).or_insert([None; 3]),
                    _ => scl.entry(ch.bone).or_insert([None; 3]),
                };
                if ch.axis < 3 {
                    slot[ch.axis] = Some(v);
                }
            }
            let arm_map = clip_tracks.entry(arm_id).or_default();
            let bones: BTreeSet<u32> = loc.keys().chain(rot.keys()).chain(scl.keys()).copied().collect();
            for bone in bones {
                // Defaults from rest model if channel missing
                let model_id = id_to_bone
                    .iter()
                    .find(|(_, b)| **b == bone)
                    .map(|(id, _)| *id);
                let model = model_id.and_then(|id| doc.models.get(&id));
                let def_t = model.map(|m| m.translation).unwrap_or([0.0; 3]);
                let def_r = model.map(|m| m.rotation_deg).unwrap_or([0.0; 3]);
                let def_s = model.map(|m| m.scale).unwrap_or([1.0; 3]);
                let lv = loc.get(&bone).copied().unwrap_or([None; 3]);
                let rv = rot.get(&bone).copied().unwrap_or([None; 3]);
                let sv = scl.get(&bone).copied().unwrap_or([None; 3]);
                // Absolute Lcl from curves → delta from rest so delta-pose bones work.
                let abs_t = Vec3::new(
                    lv[0].unwrap_or(def_t[0]) * doc.unit_to_meters,
                    lv[1].unwrap_or(def_t[1]) * doc.unit_to_meters,
                    lv[2].unwrap_or(def_t[2]) * doc.unit_to_meters,
                );
                let rest_t = Vec3::new(
                    def_t[0] * doc.unit_to_meters,
                    def_t[1] * doc.unit_to_meters,
                    def_t[2] * doc.unit_to_meters,
                );
                let translation = abs_t - rest_t;
                let euler = [
                    rv[0].unwrap_or(def_r[0]),
                    rv[1].unwrap_or(def_r[1]),
                    rv[2].unwrap_or(def_r[2]),
                ];
                let order = model.map(|m| m.rotation_order).unwrap_or(0);
                let abs_r = euler_deg_to_quat(euler, order);
                let rest_r = euler_deg_to_quat(def_r, order);
                let rotation = (rest_r.inverse() * abs_r).normalize();
                let abs_s = Vec3::new(
                    sv[0].unwrap_or(def_s[0]),
                    sv[1].unwrap_or(def_s[1]),
                    sv[2].unwrap_or(def_s[2]),
                );
                let rest_s = Vec3::new(def_s[0], def_s[1], def_s[2]);
                let scale = Vec3::new(
                    if rest_s.x.abs() > 1e-8 { abs_s.x / rest_s.x } else { abs_s.x },
                    if rest_s.y.abs() > 1e-8 { abs_s.y / rest_s.y } else { abs_s.y },
                    if rest_s.z.abs() > 1e-8 { abs_s.z / rest_s.z } else { abs_s.z },
                );
                let track = arm_map.entry(bone).or_default();
                track.insert_or_merge(frame, Some(translation), Some(rotation), Some(scale));
            }
        }
        scene.animation.bone_tracks.push(clip_tracks);
    }

    if !scene.animation.clips.is_empty() {
        scene.animation.select_clip(Some(0));
        if let Some(c) = scene.animation.clips.first() {
            scene.animation.frame_start = c.frame_start;
            scene.animation.frame_end = c.frame_end;
            scene.animation.frame_current = c.frame_start;
        }
    }
    let _ = parents;
}

fn channel_axis(prop: &str) -> usize {
    let p = prop.to_ascii_lowercase();
    if p.ends_with("|x") || p == "x" || p.ends_with(".x") {
        0
    } else if p.ends_with("|y") || p == "y" || p.ends_with(".y") {
        1
    } else if p.ends_with("|z") || p == "z" || p.ends_with(".z") {
        2
    } else if p.contains("x") && !p.contains("y") && !p.contains("z") {
        0
    } else if p.contains("y") && !p.contains("z") {
        1
    } else if p.contains("z") {
        2
    } else {
        0
    }
}

fn channel_kind(model_prop: &str) -> u8 {
    let p = model_prop.to_ascii_lowercase();
    if p.contains("translation") || p.contains("location") || p.contains("position") {
        0
    } else if p.contains("rotation") || p.contains("rotate") {
        1
    } else if p.contains("scaling") || p.contains("scale") {
        2
    } else {
        0
    }
}

fn eval_anim_curve(curve: &AnimCurveDef, time: f64) -> f32 {
    if curve.times.is_empty() || curve.values.is_empty() {
        return 0.0;
    }
    if time <= curve.times[0] {
        return curve.values[0];
    }
    let last = curve.times.len() - 1;
    if time >= curve.times[last] {
        return curve.values[last];
    }
    // Linear search is fine for typical curves; binary would be better for long takes
    for i in 0..last {
        let t0 = curve.times[i];
        let t1 = curve.times[i + 1];
        if time >= t0 && time <= t1 {
            let u = if (t1 - t0).abs() < 1e-12 {
                0.0
            } else {
                ((time - t0) / (t1 - t0)) as f32
            };
            return curve.values[i] * (1.0 - u) + curve.values[i + 1] * u;
        }
    }
    curve.values[last]
}

fn euler_deg_to_quat(euler_deg: [f32; 3], order: i32) -> Quat {
    let x = euler_deg[0].to_radians();
    let y = euler_deg[1].to_radians();
    let z = euler_deg[2].to_radians();
    let qx = Quat::from_rotation_x(x);
    let qy = Quat::from_rotation_y(y);
    let qz = Quat::from_rotation_z(z);
    // FBX RotationOrder: 0 XYZ, 1 XZY, 2 YZX, 3 YXZ, 4 ZXY, 5 ZYX
    match order {
        1 => qx * qz * qy,
        2 => qy * qz * qx,
        3 => qy * qx * qz,
        4 => qz * qx * qy,
        5 => qz * qy * qx,
        _ => qx * qy * qz, // XYZ
    }
}


/// Attach SkinBinding to mesh objects from FBX Skin/Cluster deformers.
///
/// Inverse-bind matrices use FBX Cluster `Transform` (geometry→bone) when present.
/// If they look invalid we leave `inverse_binds` empty so the engine uses
/// pose×rest⁻¹ from bone head/tail — that keeps the rest pose undeformed.
fn apply_fbx_skins(
    scene: &mut Scene,
    doc: &FbxDoc,
    parents: &HashMap<i64, i64>,
    arm_id: u64,
    mesh_ctrl_points: &HashMap<String, Vec<usize>>,
) {
    if doc.clusters.is_empty() {
        return;
    }
    let has_explicit_limb = doc.models.values().any(|m| {
        let k = m.kind.to_ascii_lowercase();
        k == "limbnode" || k == "limb" || k == "root"
    });
    // Must match the order used when building the armature.
    let limb_ids: Vec<i64> = doc
        .models
        .iter()
        .filter_map(|(&id, m)| {
            let k = m.kind.to_ascii_lowercase();
            let is_bone = k == "limbnode"
                || k == "root"
                || k == "limb"
                || k == "bone"
                || k.contains("limb");
            let is_null = has_explicit_limb && (k == "null" || k == "locator" || k == "empty");
            (is_bone || is_null).then_some(id)
        })
        .collect();
    let id_to_bone: HashMap<i64, u32> = limb_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i as u32))
        .collect();
    if id_to_bone.is_empty() {
        return;
    }
    let bone_count = limb_ids.len();

    let mut geom_skins: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut skin_clusters: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut cluster_bone: HashMap<i64, i64> = HashMap::new();
    let mut geom_model: HashMap<i64, i64> = HashMap::new();

    for c in &doc.connections {
        if c.ty != "OO" {
            continue;
        }
        if doc.skins.contains_key(&c.child) && doc.geometries.contains_key(&c.parent) {
            geom_skins.entry(c.parent).or_default().push(c.child);
        }
        if doc.skins.contains_key(&c.parent) && doc.geometries.contains_key(&c.child) {
            geom_skins.entry(c.child).or_default().push(c.parent);
        }
        if doc.clusters.contains_key(&c.child) && doc.skins.contains_key(&c.parent) {
            skin_clusters.entry(c.parent).or_default().push(c.child);
        }
        if doc.clusters.contains_key(&c.parent) && doc.skins.contains_key(&c.child) {
            skin_clusters.entry(c.child).or_default().push(c.parent);
        }
        if doc.clusters.contains_key(&c.child) && doc.models.contains_key(&c.parent) {
            cluster_bone.insert(c.child, c.parent);
        }
        if doc.clusters.contains_key(&c.parent) && doc.models.contains_key(&c.child) {
            cluster_bone.insert(c.parent, c.child);
        }
        if doc.geometries.contains_key(&c.child) && doc.models.contains_key(&c.parent) {
            geom_model.insert(c.child, c.parent);
        }
        if doc.geometries.contains_key(&c.parent) && doc.models.contains_key(&c.child) {
            geom_model.insert(c.parent, c.child);
        }
    }

    let mut name_to_obj: HashMap<String, usize> = HashMap::new();
    for (i, o) in scene.objects.iter().enumerate() {
        if !o.mesh.indices.is_empty() {
            name_to_obj.insert(o.name.to_ascii_lowercase(), i);
        }
    }

    let u = doc.unit_to_meters;

    for (gid, skin_ids) in &geom_skins {
        let Some(g) = doc.geometries.get(gid) else { continue };
        let nverts = g.vertices.len();
        if nverts == 0 {
            continue;
        }
        let model_name = geom_model
            .get(gid)
            .and_then(|mid| doc.models.get(mid))
            .map(|m| m.name.to_ascii_lowercase())
            .unwrap_or_else(|| g.name.to_ascii_lowercase());
        let Some(&obj_i) = name_to_obj.get(&model_name).or_else(|| {
            name_to_obj
                .iter()
                .find(|(n, _)| n.contains(&model_name) || model_name.contains(n.as_str()))
                .map(|(_, i)| i)
        }) else {
            continue;
        };

        // Accumulator: per control-point list of (bone, weight)
        let mut raw: Vec<Vec<(u16, f32)>> = vec![Vec::new(); nverts];
        let mut inverse_binds = vec![Mat4::IDENTITY; bone_count];
        let mut has_any_ibm = false;

        for &sid in skin_ids {
            let Some(cids) = skin_clusters.get(&sid) else { continue };
            for &cid in cids {
                let Some(cluster) = doc.clusters.get(&cid) else { continue };
                let Some(&bone_model) = cluster_bone.get(&cid) else { continue };
                let Some(&bone_idx) = id_to_bone.get(&bone_model) else { continue };
                let bi = bone_idx as usize;

                // Prefer Transform (geometry→bone). Fall back to inverse(TransformLink).
                if let Some(xf) = cluster.transform {
                    let m = mat4_from_fbx_row_major(xf, u);
                    if mat4_is_reasonable(m) {
                        inverse_binds[bi] = m;
                        has_any_ibm = true;
                    }
                } else if let Some(link) = cluster.transform_link {
                    let link_m = mat4_from_fbx_row_major(link, u);
                    if mat4_is_reasonable(link_m) {
                        inverse_binds[bi] = link_m.inverse();
                        has_any_ibm = true;
                    }
                }

                for (ii, &vi) in cluster.indexes.iter().enumerate() {
                    if vi < 0 {
                        continue;
                    }
                    let vi = vi as usize;
                    if vi >= nverts {
                        continue;
                    }
                    let w = cluster.weights.get(ii).copied().unwrap_or(0.0);
                    if w > 1e-8 {
                        raw[vi].push((bone_idx as u16, w));
                    }
                }
            }
        }

        let mut joints = vec![[0u16; 4]; nverts];
        let mut weights = vec![[0f32; 4]; nverts];
        for vi in 0..nverts {
            let mut pairs = raw[vi].clone();
            if pairs.is_empty() {
                // Unweighted control point: bind fully to root so it doesn't fly away
                pairs.push((0, 1.0));
            }
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            // Merge duplicate bones
            let mut merged: Vec<(u16, f32)> = Vec::new();
            for (j, w) in pairs {
                if let Some(slot) = merged.iter_mut().find(|(jj, _)| *jj == j) {
                    slot.1 += w;
                } else {
                    merged.push((j, w));
                }
            }
            merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            merged.truncate(4);
            let sum: f32 = merged.iter().map(|p| p.1).sum::<f32>().max(1e-8);
            for (k, (j, w)) in merged.into_iter().enumerate() {
                joints[vi][k] = j;
                weights[vi][k] = w / sum;
            }
        }

        let obj = &mut scene.objects[obj_i];
        let nv = obj.mesh.vertices.len();
        let ctrl = mesh_ctrl_points.get(&model_name);
        let (j_out, w_out) = if let Some(map) = ctrl {
            let mut j2 = vec![[0u16; 4]; nv];
            let mut w2 = vec![[0f32; 4]; nv];
            for i in 0..nv {
                let cp = map.get(i).copied().unwrap_or(0).min(nverts.saturating_sub(1));
                j2[i] = joints[cp];
                w2[i] = weights[cp];
            }
            (j2, w2)
        } else if nv == nverts {
            (joints, weights)
        } else {
            (vec![[0, 0, 0, 0]; nv], vec![[1.0, 0.0, 0.0, 0.0]; nv])
        };

        // Keep the imported mesh transform so the bind-pose silhouette matches the file /
        // GLB. Empty IBM keeps LBS at identity while pose is rest, so this does not stretch.
        obj.parent = Some(arm_id);

        // Rebuild rest from current (undeformed) mesh vertices.
        let rest_p: Vec<[f32; 3]> = obj.mesh.vertices.iter().map(|v| v.position).collect();
        let rest_n: Vec<[f32; 3]> = obj.mesh.vertices.iter().map(|v| v.normal).collect();

        // Never use FBX Cluster Transform as IBM here: wrong layout/space is what
        // stretched meshes into spaghetti. Empty IBM → engine uses pose×rest⁻¹
        // which is identity at rest (undeformed mesh). Weights still drive pose.
        obj.skin = Some(SkinBinding {
            armature_id: arm_id,
            joints: j_out,
            weights: w_out,
            rest_positions: rest_p,
            rest_normals: rest_n,
            inverse_binds: Vec::new(),
        });
    }
    let _ = parents;
}

fn mat4_from_fbx_row_major(a: [f32; 16], unit: f32) -> Mat4 {
    // FBX dumps matrices row-major:
    //   [m00 m01 m02 m03 | m10 m11 m12 m13 | m20 m21 m22 m23 | m30 m31 m32 m33]
    // Translation is (m03, m13, m23) → indices 3, 7, 11. glam columns are
    // (m00,m10,m20,m30), (m01,m11,m21,m31), …
    Mat4::from_cols(
        glam::Vec4::new(a[0], a[4], a[8], a[12]),
        glam::Vec4::new(a[1], a[5], a[9], a[13]),
        glam::Vec4::new(a[2], a[6], a[10], a[14]),
        glam::Vec4::new(a[3] * unit, a[7] * unit, a[11] * unit, a[15]),
    )
}

fn mat4_is_reasonable(m: Mat4) -> bool {
    let a = m.to_cols_array();
    if a.iter().any(|v| !v.is_finite()) {
        return false;
    }
    // Reject near-zero scale (would explode vertices)
    let sx = Vec3::new(a[0], a[1], a[2]).length();
    let sy = Vec3::new(a[4], a[5], a[6]).length();
    let sz = Vec3::new(a[8], a[9], a[10]).length();
    sx > 1e-5 && sy > 1e-5 && sz > 1e-5 && sx < 1e5 && sy < 1e5 && sz < 1e5
}


/// Read one layer-element value (normal / uv) honouring FBX mapping + reference modes.
#[allow(clippy::too_many_arguments)]
fn layer_value<T: Copy>(
    data: &[T],
    indices: &[i32],
    mapping: &str,
    reference: &str,
    corner: usize,
    vertex: usize,
    polygon: usize,
    n_corners: usize,
    n_vertices: usize,
) -> Option<T> {
    let slot = match mapping {
        "ByPolygonVertex" => corner,
        "ByVertice" | "ByVertex" => vertex,
        "ByPolygon" => polygon,
        "AllSame" => 0,
        // missing/unknown mapping: guess from the array length
        _ => {
            let n = if indices.is_empty() { data.len() } else { indices.len() };
            if n == n_corners {
                corner
            } else if n == n_vertices {
                vertex
            } else {
                corner
            }
        }
    };
    if (reference == "IndexToDirect" || reference == "Index") || (reference.is_empty() && !indices.is_empty()) {
        let i = *indices.get(slot)?;
        if i < 0 {
            return None;
        }
        data.get(i as usize).copied()
    } else {
        data.get(slot).copied()
    }
}

/// Triangulate the polygons and resolve per-corner normals / UVs. Returns the mesh and the
/// material slot of each triangle.
fn geom_to_mesh(g: &Geom) -> Result<(MeshData, Vec<u32>, Vec<usize>)> {
    let nv = g.vertices.len();
    // polygon corners: (position index, polygon number)
    let mut corners: Vec<(usize, usize)> = Vec::with_capacity(g.polygon_vertex_index.len());
    let mut polys: Vec<(usize, usize)> = Vec::new(); // (first corner, corner count)
    let mut start = 0usize;
    for &idx in &g.polygon_vertex_index {
        let last = idx < 0;
        let v = if last { !idx } else { idx };
        corners.push((v.max(0) as usize, polys.len()));
        if last {
            polys.push((start, corners.len() - start));
            start = corners.len();
        }
    }
    let ncorners = corners.len();

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut weld: HashMap<(usize, [u32; 3], [u32; 2]), u32> = HashMap::new();
    let mut corner_vertex: Vec<u32> = vec![0; ncorners];
    for (k, &(vi, poly)) in corners.iter().enumerate() {
        if vi >= nv {
            continue; // broken polygon is dropped below
        }
        let normal = layer_value(
            &g.normals,
            &g.normal_indices,
            &g.normal_mapping,
            &g.normal_ref,
            k,
            vi,
            poly,
            ncorners,
            nv,
        )
        .unwrap_or([0.0; 3]);
        let uv = layer_value(
            &g.uvs,
            &g.uv_indices,
            &g.uv_mapping,
            &g.uv_ref,
            k,
            vi,
            poly,
            ncorners,
            nv,
        )
        .map(|u| [u[0], 1.0 - u[1]])
        .unwrap_or([0.0; 2]);
        let key = (vi, normal.map(f32::to_bits), uv.map(f32::to_bits));
        let id = *weld.entry(key).or_insert_with(|| {
            vertices.push(Vertex {
                position: g.vertices[vi],
                normal,
                uv,
                color: [1.0, 1.0, 1.0, 1.0],
            });
            (vertices.len() - 1) as u32
        });
        corner_vertex[k] = id;
    }
    // Parallel to `vertices`: FBX control-point index for skin cluster Indexes.
    let mut ctrl_of_vertex = vec![0usize; vertices.len()];
    for ((vi, _, _), &vid) in weld.iter() {
        if (vid as usize) < ctrl_of_vertex.len() {
            ctrl_of_vertex[vid as usize] = *vi;
        }
    }

    let mut indices: Vec<u32> = Vec::new();
    let mut tri_slot: Vec<u32> = Vec::new();
    for (pi, &(first, count)) in polys.iter().enumerate() {
        if count < 3 || corners[first..first + count].iter().any(|c| c.0 >= nv) {
            continue;
        }
        let slot = match g.material_mapping.as_str() {
            "ByPolygon" => g.material_ids.get(pi).copied().unwrap_or(0),
            _ => g.material_ids.first().copied().unwrap_or(0),
        }
        .max(0) as u32;
        // fan triangulation
        for i in 1..count - 1 {
            indices.push(corner_vertex[first]);
            indices.push(corner_vertex[first + i]);
            indices.push(corner_vertex[first + i + 1]);
            tri_slot.push(slot);
        }
    }

    let has_normals = !g.normals.is_empty();
    let mut md = MeshData {
        name: g.name.clone(),
        vertices,
        indices,
        groups: Vec::new(),
        ..Default::default()
    };
    if !has_normals {
        md.compute_normals();
    }
    Ok((md, tri_slot, ctrl_of_vertex))
}

// ---------------------------------------------------------------------------
// Exporter (local ASCII FBX 7.4)
// ---------------------------------------------------------------------------

fn build_ascii_fbx(scene: &Scene, only: Option<&BTreeSet<u64>>) -> Result<String> {
    let mut out = String::new();
    out.push_str("; FBX 7.4.0 project file\n; Created by BlendCraft (local ASCII exporter)\n\n");
    out.push_str("FBXHeaderExtension:  {\n");
    out.push_str("    FBXHeaderVersion: 1003\n");
    out.push_str("    FBXVersion: 7400\n");
    out.push_str("}\n\n");
    out.push_str("GlobalSettings:  {\n");
    out.push_str("    Version: 1000\n");
    out.push_str("    Properties70:  {\n");
    out.push_str("        P: \"UpAxis\", \"int\", \"Integer\", \"\",1\n");
    out.push_str("        P: \"UpAxisSign\", \"int\", \"Integer\", \"\",1\n");
    out.push_str("        P: \"FrontAxis\", \"int\", \"Integer\", \"\",2\n");
    out.push_str("        P: \"FrontAxisSign\", \"int\", \"Integer\", \"\",1\n");
    out.push_str("        P: \"CoordAxis\", \"int\", \"Integer\", \"\",0\n");
    out.push_str("        P: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1\n");
    out.push_str("        P: \"UnitScaleFactor\", \"double\", \"Number\", \"\",100\n");
    out.push_str("        P: \"OriginalUnitScaleFactor\", \"double\", \"Number\", \"\",100\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");
    out.push_str("Definitions:  {\n    Version: 100\n    Count: 3\n");
    out.push_str("    ObjectType: \"Geometry\" {\n        Count: 1\n    }\n");
    out.push_str("    ObjectType: \"Model\" {\n        Count: 1\n    }\n");
    out.push_str("    ObjectType: \"Material\" {\n        Count: 1\n    }\n}\n\n");
    out.push_str("Objects:  {\n");

    let mut next_id: i64 = 100000;
    let mut mat_ids: Vec<i64> = Vec::new();
    let mut model_links: Vec<(i64, i64, usize)> = Vec::new(); // model, geom, mat_index

    for m in &scene.materials {
        let id = next_id;
        next_id += 1;
        mat_ids.push(id);
        let c = m.base_color;
        out.push_str(&format!(
            "    Material: {id}, \"Material::{}\", \"\" {{\n",
            sanitize(&m.name)
        ));
        out.push_str("        Version: 102\n");
        out.push_str("        ShadingModel: \"Phong\"\n");
        out.push_str("        MultiLayer: 0\n");
        out.push_str(&format!(
            "        P: \"DiffuseColor\", \"Color\", \"\", \"A\",{:.6},{:.6},{:.6}\n",
            c[0], c[1], c[2]
        ));
        out.push_str(&format!(
            "        P: \"Diffuse\", \"Number\", \"\", \"A\",{:.6}\n",
            (c[0] + c[1] + c[2]) / 3.0
        ));
        out.push_str("    }\n");
    }
    if mat_ids.is_empty() {
        let id = next_id;
        next_id += 1;
        mat_ids.push(id);
        out.push_str(&format!(
            "    Material: {id}, \"Material::Default\", \"\" {{\n        Version: 102\n        ShadingModel: \"Phong\"\n        P: \"DiffuseColor\", \"Color\", \"\", \"A\",0.8,0.8,0.8\n    }}\n"
        ));
    }

    for obj in &scene.objects {
        if let Some(sel) = only {
            if !sel.contains(&obj.id) {
                continue;
            }
        }
        match &obj.kind {
            ObjectKind::Mesh => {}
            _ => continue, // non-mesh helpers (surface, volume, armature, etc.)
        }

        let geom_id = next_id;
        next_id += 1;
        let model_id = next_id;
        next_id += 1;
        model_links.push((model_id, geom_id, obj.material.unwrap_or(0)));

        // Bake transform into vertices for maximum compatibility
        let world = Mat4::from_scale_rotation_translation(obj.scale, obj.rotation, obj.translation);
        let baked = obj.mesh.transformed(world);

        let mut verts = String::new();
        for (i, v) in baked.vertices.iter().enumerate() {
            if i > 0 {
                verts.push(',');
            }
            verts.push_str(&format!(
                "{:.6},{:.6},{:.6}",
                v.position[0], v.position[1], v.position[2]
            ));
        }
        let mut pvi = String::new();
        for (i, tri) in baked.indices.chunks_exact(3).enumerate() {
            if i > 0 {
                pvi.push(',');
            }
            let a = tri[0] as i32;
            let b = tri[1] as i32;
            let c = -(tri[2] as i32) - 1;
            pvi.push_str(&format!("{a},{b},{c}"));
        }
        // normals
        let mut norms = String::new();
        for (i, v) in baked.vertices.iter().enumerate() {
            if i > 0 {
                norms.push(',');
            }
            norms.push_str(&format!(
                "{:.6},{:.6},{:.6}",
                v.normal[0], v.normal[1], v.normal[2]
            ));
        }
        let vcount = baked.vertices.len() * 3;
        let icount = baked.indices.len();

        out.push_str(&format!(
            "    Geometry: {geom_id}, \"Geometry::{}\", \"Mesh\" {{\n",
            sanitize(&obj.mesh.name)
        ));
        out.push_str(&format!(
            "        Vertices: *{vcount} {{\n            a: {verts}\n        }}\n"
        ));
        out.push_str(&format!(
            "        PolygonVertexIndex: *{icount} {{\n            a: {pvi}\n        }}\n"
        ));
        out.push_str("        GeometryVersion: 124\n");
        out.push_str("        LayerElementNormal: 0 {\n");
        out.push_str("            Version: 101\n");
        out.push_str("            Name: \"\"\n");
        out.push_str("            MappingInformationType: \"ByVertice\"\n");
        out.push_str("            ReferenceInformationType: \"Direct\"\n");
        out.push_str(&format!(
            "            Normals: *{vcount} {{\n                a: {norms}\n            }}\n"
        ));
        out.push_str("        }\n");
        out.push_str("        Layer: 0 {\n            Version: 100\n");
        out.push_str("            LayerElement:  {\n                Type: \"LayerElementNormal\"\n                TypedIndex: 0\n            }\n");
        out.push_str("        }\n");
        out.push_str("    }\n");

        out.push_str(&format!(
            "    Model: {model_id}, \"Model::{}\", \"Mesh\" {{\n",
            sanitize(&obj.name)
        ));
        out.push_str("        Version: 232\n");
        out.push_str("        Properties70:  {\n");
        out.push_str("            P: \"InheritType\", \"enum\", \"\", \"\",1\n");
        out.push_str("            P: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",0\n");
        // Identity transform (already baked)
        out.push_str(
            "            P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,0,0\n",
        );
        out.push_str("            P: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\",0,0,0\n");
        out.push_str("            P: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A\",1,1,1\n");
        out.push_str("        }\n");
        out.push_str("        Shading: Y\n");
        out.push_str("        Culling: \"CullingOff\"\n");
        out.push_str("    }\n");
    }

    out.push_str("}\n\n");
    out.push_str("Connections:  {\n");
    for &(model_id, geom_id, mat_i) in &model_links {
        out.push_str(&format!("    C: \"OO\",{geom_id},{model_id}\n"));
        out.push_str(&format!("    C: \"OO\",{model_id},0\n"));
        let mid = mat_ids
            .get(mat_i)
            .or_else(|| mat_ids.first())
            .copied()
            .unwrap_or(0);
        if mid != 0 {
            out.push_str(&format!("    C: \"OO\",{mid},{model_id}\n"));
        }
    }
    out.push_str("}\n");

    if model_links.is_empty() {
        bail!("nothing to export as FBX (no mesh objects)");
    }
    Ok(out)
}

fn sanitize(s: &str) -> String {
    s.replace('\"', "").replace('\n', " ")
}

// ===========================================================================
// Binary FBX parser (local, no external FBX libraries)
// ===========================================================================

struct BinReader<'a> {
    data: &'a [u8],
    pos: usize,
    version: u32,
}

impl<'a> BinReader<'a> {
    fn new(data: &'a [u8], version: u32) -> Self {
        Self {
            data,
            pos: 0,
            version,
        }
    }
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
    fn u8(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            bail!("unexpected end of FBX");
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32> {
        if self.pos + 4 > self.data.len() {
            bail!("unexpected end of FBX");
        }
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn u64(&mut self) -> Result<u64> {
        if self.pos + 8 > self.data.len() {
            bail!("unexpected end of FBX");
        }
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f64(&mut self) -> Result<f64> {
        if self.pos + 8 > self.data.len() {
            bail!("unexpected end of FBX");
        }
        let v = f64::from_bits(u64::from_le_bytes(
            self.data[self.pos..self.pos + 8].try_into().unwrap(),
        ));
        self.pos += 8;
        Ok(v)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            bail!("unexpected end of FBX");
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn offset_size(&self) -> usize {
        if self.version >= 7500 {
            8
        } else {
            4
        }
    }
    fn read_offset(&mut self) -> Result<u64> {
        if self.version >= 7500 {
            self.u64()
        } else {
            Ok(self.u32()? as u64)
        }
    }
    fn null_record_size(&self) -> usize {
        // EndOffset + NumProps + PropListLen + NameLen(1)
        if self.version >= 7500 {
            8 + 8 + 8 + 1
        } else {
            4 + 4 + 4 + 1
        }
    }
}

#[derive(Debug)]
struct BinNode {
    name: String,
    props: Vec<BinProp>,
    children: Vec<BinNode>,
}

#[derive(Debug, Clone)]
enum BinProp {
    Bool(bool),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    String(String),
    Bytes(Vec<u8>),
    F32Arr(Vec<f32>),
    F64Arr(Vec<f64>),
    I32Arr(Vec<i32>),
    I64Arr(Vec<i64>),
}

fn parse_binary(data: &[u8]) -> Result<FbxDoc> {
    if data.len() < 27 {
        bail!("file too small for binary FBX");
    }
    // Header: "Kaydara FBX Binary  \x00" (21) + 0x1A 0x00 + version u32
    let version = u32::from_le_bytes(data[23..27].try_into().unwrap());
    let _ = version; // accept common 7.x versions; still attempt others
    let mut r = BinReader::new(data, version);
    r.pos = 27;

    let mut roots = Vec::new();
    while r.pos + r.null_record_size() <= data.len() {
        // Peek null record
        let peek = &data[r.pos..];
        let nsz = r.null_record_size();
        if peek.len() >= nsz && peek[..nsz].iter().all(|&b| b == 0) {
            r.pos += nsz;
            break;
        }
        match read_node(&mut r) {
            Ok(Some(node)) => roots.push(node),
            Ok(None) => break,
            // Keep what was read so far (e.g. a broken trailing footer section).
            Err(_) if roots.iter().any(|n| n.name == "Objects") => break,
            Err(e) => return Err(e),
        }
    }

    let mut doc = FbxDoc::default();
    for root in &roots {
        collect_from_node(root, &mut doc);
    }
    // Also walk Objects/Connections specifically
    for root in &roots {
        if root.name == "GlobalSettings" {
            fill_global_settings(root, &mut doc);
        }
        if root.name == "Objects" {
            for ch in &root.children {
                ingest_object(ch, &mut doc);
            }
        }
        if root.name == "Connections" {
            for ch in &root.children {
                if ch.name == "C" {
                    parse_bin_connection(ch, &mut doc);
                }
            }
        }
    }
    if doc.geometries.is_empty() {
        bail!("Binary FBX: no polygon mesh Geometry found (version {version})");
    }
    Ok(doc)
}

fn read_node(r: &mut BinReader<'_>) -> Result<Option<BinNode>> {
    if r.remaining() < r.null_record_size() {
        return Ok(None);
    }
    let start = r.pos;
    let end_offset = r.read_offset()?;
    let num_props = r.read_offset()?;
    let _prop_list_len = r.read_offset()?;
    let name_len = r.u8()? as usize;

    // NULL record
    if end_offset == 0 && num_props == 0 && name_len == 0 {
        return Ok(None);
    }

    let name = if name_len > 0 {
        String::from_utf8_lossy(r.bytes(name_len)?).into_owned()
    } else {
        String::new()
    };

    let mut props = Vec::with_capacity(num_props as usize);
    for _ in 0..num_props {
        props.push(read_prop(r)?);
    }

    let mut children = Vec::new();
    // Nested list ends at end_offset
    let end = end_offset as usize;
    if end > r.data.len() {
        bail!("FBX node end offset past EOF for '{}'", name);
    }
    while r.pos + r.null_record_size() <= end {
        let save = r.pos;
        // check null
        let nsz = r.null_record_size();
        if r.data[r.pos..r.pos + nsz].iter().all(|&b| b == 0) {
            r.pos += nsz;
            break;
        }
        match read_node(r) {
            Ok(Some(n)) => children.push(n),
            Ok(None) => break,
            Err(_) => {
                // Unreadable child (unknown property type, truncated array...): skip the rest
                // of this node instead of failing the whole file.
                r.pos = end;
                break;
            }
        }
        if r.pos == save {
            break; // safety
        }
    }
    // Ensure we land on end_offset
    if end_offset > 0 && (r.pos as u64) < end_offset {
        r.pos = end_offset as usize;
    }
    let _ = start;
    Ok(Some(BinNode {
        name,
        props,
        children,
    }))
}

fn read_prop(r: &mut BinReader<'_>) -> Result<BinProp> {
    let t = r.u8()? as char;
    match t {
        'Y' => {
            let v = r.u8()? as i16 | ((r.u8()? as i16) << 8);
            Ok(BinProp::I16(v))
        }
        'C' => Ok(BinProp::Bool(r.u8()? != 0)),
        'I' => Ok(BinProp::I32(r.i32()?)),
        'F' => Ok(BinProp::F32(r.f32()?)),
        'D' => Ok(BinProp::F64(r.f64()?)),
        'L' => {
            let v = r.u64()? as i64;
            Ok(BinProp::I64(v))
        }
        'S' | 'R' => {
            let len = r.u32()? as usize;
            let b = r.bytes(len)?.to_vec();
            if t == 'S' {
                Ok(BinProp::String(String::from_utf8_lossy(&b).into_owned()))
            } else {
                Ok(BinProp::Bytes(b))
            }
        }
        'f' => Ok(BinProp::F32Arr(read_array_f32(r)?)),
        'd' => Ok(BinProp::F64Arr(read_array_f64(r)?)),
        'i' => Ok(BinProp::I32Arr(read_array_i32(r)?)),
        'l' => Ok(BinProp::I64Arr(read_array_i64(r)?)),
        'b' => {
            let raw = read_array_raw(r)?;
            Ok(BinProp::Bytes(raw))
        }
        other => bail!("unknown FBX property type '{other}'"),
    }
}

fn read_array_raw(r: &mut BinReader<'_>) -> Result<Vec<u8>> {
    let length = r.u32()? as usize;
    let encoding = r.u32()?;
    let comp_len = r.u32()? as usize;
    let data = r.bytes(comp_len)?.to_vec();
    if encoding == 0 {
        if data.len() < length {
            // length is element count for typed arrays; for raw it's bytes
            return Ok(data);
        }
        Ok(data)
    } else if encoding == 1 {
        use flate2::read::ZlibDecoder;
        use std::io::Read;
        let mut dec = ZlibDecoder::new(&data[..]);
        let mut out = Vec::new();
        dec.read_to_end(&mut out)
            .context("zlib decompress FBX array")?;
        Ok(out)
    } else {
        bail!("unsupported FBX array encoding {encoding}")
    }
}

fn read_array_f32(r: &mut BinReader<'_>) -> Result<Vec<f32>> {
    let length = r.u32()? as usize;
    let encoding = r.u32()?;
    let comp_len = r.u32()? as usize;
    let data = r.bytes(comp_len)?.to_vec();
    let raw = if encoding == 1 {
        use flate2::read::ZlibDecoder;
        use std::io::Read;
        let mut dec = ZlibDecoder::new(&data[..]);
        let mut out = Vec::new();
        dec.read_to_end(&mut out).context("zlib f32")?;
        out
    } else {
        data
    };
    let mut v = Vec::with_capacity(length);
    for chunk in raw.chunks_exact(4).take(length) {
        v.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(v)
}

fn read_array_f64(r: &mut BinReader<'_>) -> Result<Vec<f64>> {
    let length = r.u32()? as usize;
    let encoding = r.u32()?;
    let comp_len = r.u32()? as usize;
    let data = r.bytes(comp_len)?.to_vec();
    let raw = if encoding == 1 {
        use flate2::read::ZlibDecoder;
        use std::io::Read;
        let mut dec = ZlibDecoder::new(&data[..]);
        let mut out = Vec::new();
        dec.read_to_end(&mut out).context("zlib f64")?;
        out
    } else {
        data
    };
    let mut v = Vec::with_capacity(length);
    for chunk in raw.chunks_exact(8).take(length) {
        v.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(v)
}

fn read_array_i32(r: &mut BinReader<'_>) -> Result<Vec<i32>> {
    let length = r.u32()? as usize;
    let encoding = r.u32()?;
    let comp_len = r.u32()? as usize;
    let data = r.bytes(comp_len)?.to_vec();
    let raw = if encoding == 1 {
        use flate2::read::ZlibDecoder;
        use std::io::Read;
        let mut dec = ZlibDecoder::new(&data[..]);
        let mut out = Vec::new();
        dec.read_to_end(&mut out).context("zlib i32")?;
        out
    } else {
        data
    };
    let mut v = Vec::with_capacity(length);
    for chunk in raw.chunks_exact(4).take(length) {
        v.push(i32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(v)
}

fn read_array_i64(r: &mut BinReader<'_>) -> Result<Vec<i64>> {
    let length = r.u32()? as usize;
    let encoding = r.u32()?;
    let comp_len = r.u32()? as usize;
    let data = r.bytes(comp_len)?.to_vec();
    let raw = if encoding == 1 {
        use flate2::read::ZlibDecoder;
        use std::io::Read;
        let mut dec = ZlibDecoder::new(&data[..]);
        let mut out = Vec::new();
        dec.read_to_end(&mut out).context("zlib i64")?;
        out
    } else {
        data
    };
    let mut v = Vec::with_capacity(length);
    for chunk in raw.chunks_exact(8).take(length) {
        v.push(i64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(v)
}

fn collect_from_node(node: &BinNode, _doc: &mut FbxDoc) {
    // reserved for global properties if needed
    let _ = node;
}

fn ingest_object(node: &BinNode, doc: &mut FbxDoc) {
    match node.name.as_str() {
        "Geometry" => {
            let id = prop_i64(node.props.get(0)).unwrap_or(0);
            let name = prop_string(node.props.get(1)).unwrap_or_default();
            let class = prop_string(node.props.get(2)).unwrap_or_default();
            // Blend-shape targets ("Shape"), curves, NURBS... also live under Geometry but
            // are not polygon meshes.
            if !class.is_empty() && class != "Mesh" {
                return;
            }
            let mut g = Geom {
                name: strip_name(&name),
                ..Default::default()
            };
            fill_geom_from_children(node, &mut g);
            if !g.vertices.is_empty() && !g.polygon_vertex_index.is_empty() {
                doc.geometries.insert(id, g);
            }
        }
        "Model" => {
            let id = prop_i64(node.props.get(0)).unwrap_or(0);
            let name = prop_string(node.props.get(1)).unwrap_or_default();
            let class = prop_string(node.props.get(2)).unwrap_or_default();
            let mut m = Model {
                name: strip_name(&name),
                kind: class,
                scale: [1.0, 1.0, 1.0],
                ..Default::default()
            };
            fill_model_from_children(node, &mut m);
            doc.models.insert(id, m);
        }
        "Material" => {
            let id = prop_i64(node.props.get(0)).unwrap_or(0);
            let name = prop_string(node.props.get(1)).unwrap_or_default();
            let mut mat = Mat::new(strip_name(&name));
            fill_material_from_children(node, &mut mat);
            doc.materials.insert(id, mat);
        }
        "Texture" => {
            let id = prop_i64(node.props.get(0)).unwrap_or(0);
            let name = prop_string(node.props.get(1)).unwrap_or_default();
            let mut t = Tex {
                name: strip_name(&name),
                ..Default::default()
            };
            for ch in &node.children {
                if ch.name.eq_ignore_ascii_case("FileName") {
                    t.file = prop_string(ch.props.first()).unwrap_or_default();
                } else if ch.name.eq_ignore_ascii_case("RelativeFilename") {
                    t.relative = prop_string(ch.props.first()).unwrap_or_default();
                }
            }
            doc.textures.insert(id, t);
        }
        "Deformer" => {
            let id = prop_i64(node.props.get(0)).unwrap_or(0);
            let name = prop_string(node.props.get(1)).unwrap_or_default();
            let class = prop_string(node.props.get(2)).unwrap_or_default();
            let class_l = class.to_ascii_lowercase();
            if class_l == "skin" || class_l.contains("skin") {
                doc.skins.insert(id, SkinDef { name: strip_name(&name) });
            } else if class_l == "cluster" || class_l.contains("cluster") {
                let mut c = ClusterDef {
                    name: strip_name(&name),
                    ..Default::default()
                };
                for ch in &node.children {
                    match ch.name.as_str() {
                        "Indexes" | "Indices" => {
                            if let Some(BinProp::I32Arr(a)) = ch.props.first() {
                                c.indexes = a.clone();
                            } else if let Some(BinProp::I64Arr(a)) = ch.props.first() {
                                c.indexes = a.iter().map(|&x| x as i32).collect();
                            }
                        }
                        "Weights" => {
                            if let Some(BinProp::F64Arr(a)) = ch.props.first() {
                                c.weights = a.iter().map(|&x| x as f32).collect();
                            } else if let Some(BinProp::F32Arr(a)) = ch.props.first() {
                                c.weights = a.clone();
                            }
                        }
                        "Transform" => {
                            c.transform = bin_mat4(ch.props.first());
                        }
                        "TransformLink" => {
                            c.transform_link = bin_mat4(ch.props.first());
                        }
                        _ => {}
                    }
                }
                doc.clusters.insert(id, c);
            }
        }
        "AnimationStack" => {
            let id = prop_i64(node.props.get(0)).unwrap_or(0);
            let name = prop_string(node.props.get(1)).unwrap_or_default();
            let mut s = AnimStackDef {
                name: strip_name(&name),
                ..Default::default()
            };
            for ch in &node.children {
                if ch.name == "Properties70" || ch.name == "Properties60" {
                    for p in &ch.children {
                        if p.name != "P" {
                            continue;
                        }
                        let key = prop_string(p.props.get(0)).unwrap_or_default();
                        // LocalStart / LocalStop are KTime (int64) in some writers; also as double.
                        if key == "LocalStart" || key == "ReferenceStart" {
                            s.local_start = prop_ktime(p);
                        } else if key == "LocalStop" || key == "ReferenceStop" {
                            s.local_stop = prop_ktime(p);
                        }
                    }
                }
            }
            doc.anim_stacks.insert(id, s);
        }
        "AnimationLayer" => {
            let id = prop_i64(node.props.get(0)).unwrap_or(0);
            let name = prop_string(node.props.get(1)).unwrap_or_default();
            doc.anim_layers.insert(id, AnimLayerDef { name: strip_name(&name) });
        }
        "AnimationCurveNode" => {
            let id = prop_i64(node.props.get(0)).unwrap_or(0);
            let name = prop_string(node.props.get(1)).unwrap_or_default();
            let mut n = AnimCurveNodeDef {
                name: strip_name(&name),
                defaults: [0.0, 0.0, 0.0],
            };
            for ch in &node.children {
                if ch.name == "Properties70" || ch.name == "Properties60" {
                    for p in &ch.children {
                        if p.name != "P" {
                            continue;
                        }
                        let key = prop_string(p.props.get(0)).unwrap_or_default();
                        let nums = prop_trailing_f32(p);
                        let v = nums.last().copied().unwrap_or(0.0);
                        match key.as_str() {
                            "d|X" | "X" => n.defaults[0] = v,
                            "d|Y" | "Y" => n.defaults[1] = v,
                            "d|Z" | "Z" => n.defaults[2] = v,
                            _ => {}
                        }
                    }
                }
            }
            doc.anim_curve_nodes.insert(id, n);
        }
        "AnimationCurve" => {
            let id = prop_i64(node.props.get(0)).unwrap_or(0);
            let name = prop_string(node.props.get(1)).unwrap_or_default();
            let mut c = AnimCurveDef {
                name: strip_name(&name),
                ..Default::default()
            };
            for ch in &node.children {
                match ch.name.as_str() {
                    "KeyTime" => {
                        if let Some(BinProp::I64Arr(a)) = ch.props.first() {
                            c.times = a.iter().map(|&k| k as f64 / FBX_TIME_SECOND).collect();
                        } else if let Some(BinProp::I32Arr(a)) = ch.props.first() {
                            c.times = a.iter().map(|&k| k as f64 / FBX_TIME_SECOND).collect();
                        }
                    }
                    "KeyValueFloat" => {
                        if let Some(BinProp::F32Arr(a)) = ch.props.first() {
                            c.values = a.clone();
                        } else if let Some(BinProp::F64Arr(a)) = ch.props.first() {
                            c.values = a.iter().map(|&x| x as f32).collect();
                        }
                    }
                    _ => {}
                }
            }
            // Align lengths
            let n = c.times.len().min(c.values.len());
            c.times.truncate(n);
            c.values.truncate(n);
            doc.anim_curves.insert(id, c);
        }
        "Video" => {
            let id = prop_i64(node.props.get(0)).unwrap_or(0);
            let mut v = Video::default();
            for ch in &node.children {
                if ch.name.eq_ignore_ascii_case("FileName") {
                    v.file = prop_string(ch.props.first()).unwrap_or_default();
                } else if ch.name.eq_ignore_ascii_case("RelativeFilename") {
                    v.relative = prop_string(ch.props.first()).unwrap_or_default();
                } else if ch.name == "Content" {
                    if let Some(BinProp::Bytes(b)) = ch.props.first() {
                        v.content = b.clone();
                    }
                }
            }
            doc.videos.insert(id, v);
        }
        _ => {
            for ch in &node.children {
                ingest_object(ch, doc);
            }
        }
    }
}

fn bin_mat4(p: Option<&BinProp>) -> Option<[f32; 16]> {
    match p {
        Some(BinProp::F64Arr(a)) if a.len() >= 16 => {
            let mut m = [0f32; 16];
            for i in 0..16 {
                m[i] = a[i] as f32;
            }
            Some(m)
        }
        Some(BinProp::F32Arr(a)) if a.len() >= 16 => {
            let mut m = [0f32; 16];
            m.copy_from_slice(&a[..16]);
            Some(m)
        }
        _ => None,
    }
}

fn bin_f32_triplets(p: Option<&BinProp>, n: usize) -> Vec<Vec<f32>> {
    match p {
        Some(BinProp::F64Arr(a)) => a
            .chunks_exact(n)
            .map(|c| c.iter().map(|&x| x as f32).collect())
            .collect(),
        Some(BinProp::F32Arr(a)) => a.chunks_exact(n).map(|c| c.to_vec()).collect(),
        _ => Vec::new(),
    }
}

fn fill_geom_from_children(node: &BinNode, g: &mut Geom) {
    for ch in &node.children {
        match ch.name.as_str() {
            "Vertices" => {
                if let Some(BinProp::F64Arr(a)) = ch.props.first() {
                    for c in a.chunks_exact(3) {
                        g.vertices.push([c[0] as f32, c[1] as f32, c[2] as f32]);
                    }
                } else if let Some(BinProp::F32Arr(a)) = ch.props.first() {
                    for c in a.chunks_exact(3) {
                        g.vertices.push([c[0], c[1], c[2]]);
                    }
                }
            }
            "PolygonVertexIndex" => {
                if let Some(BinProp::I32Arr(a)) = ch.props.first() {
                    g.polygon_vertex_index = a.clone();
                }
            }
            "LayerElementNormal" => {
                for n in &ch.children {
                    match n.name.as_str() {
                        "Normals" => {
                            for c in bin_f32_triplets(n.props.first(), 3) {
                                g.normals.push([c[0], c[1], c[2]]);
                            }
                        }
                        "NormalsIndex" => {
                            if let Some(BinProp::I32Arr(a)) = n.props.first() {
                                g.normal_indices = a.clone();
                            }
                        }
                        "MappingInformationType" => {
                            g.normal_mapping = prop_string(n.props.first()).unwrap_or_default()
                        }
                        "ReferenceInformationType" => {
                            g.normal_ref = prop_string(n.props.first()).unwrap_or_default()
                        }
                        _ => {}
                    }
                }
            }
            "LayerElementUV" => {
                // Only the first UV set (a mesh can carry several LayerElementUV blocks).
                if !g.uvs.is_empty() {
                    continue;
                }
                for n in &ch.children {
                    match n.name.as_str() {
                        "UV" => {
                            for c in bin_f32_triplets(n.props.first(), 2) {
                                g.uvs.push([c[0], c[1]]);
                            }
                        }
                        "UVIndex" => {
                            if let Some(BinProp::I32Arr(a)) = n.props.first() {
                                g.uv_indices = a.clone();
                            }
                        }
                        "MappingInformationType" => {
                            g.uv_mapping = prop_string(n.props.first()).unwrap_or_default()
                        }
                        "ReferenceInformationType" => {
                            g.uv_ref = prop_string(n.props.first()).unwrap_or_default()
                        }
                        _ => {}
                    }
                }
            }
            "LayerElementMaterial" => {
                for n in &ch.children {
                    match n.name.as_str() {
                        "Materials" => {
                            if let Some(BinProp::I32Arr(a)) = n.props.first() {
                                g.material_ids = a.clone();
                            }
                        }
                        "MappingInformationType" => {
                            g.material_mapping =
                                prop_string(n.props.first()).unwrap_or_default()
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}

fn fill_model_from_children(node: &BinNode, m: &mut Model) {
    for ch in &node.children {
        if ch.name == "Properties70" || ch.name == "Properties60" {
            for p in &ch.children {
                if p.name != "P" {
                    continue;
                }
                let key = prop_string(p.props.get(0)).unwrap_or_default();
                let nums = prop_trailing_f32(p);
                if key == "RotationOrder" {
                    if let Some(&o) = nums.last() {
                        m.rotation_order = o as i32;
                    }
                }
                if nums.len() >= 3 {
                    let v = [
                        nums[nums.len() - 3],
                        nums[nums.len() - 2],
                        nums[nums.len() - 1],
                    ];
                    if key == "PreRotation" {
                        m.pre_rotation_deg = v;
                    } else if key == "PostRotation" {
                        m.post_rotation_deg = v;
                    } else if key.contains("Lcl Translation") {
                        m.translation = v;
                    } else if key.contains("Lcl Rotation") {
                        m.rotation_deg = v;
                    } else if key.contains("Lcl Scaling") {
                        m.scale = v;
                    } else if key.contains("GeometricTranslation") {
                        m.geom_translation = v;
                    } else if key.contains("GeometricRotation") {
                        m.geom_rotation_deg = v;
                    } else if key.contains("GeometricScaling") {
                        m.geom_scale = v;
                    }
                }
            }
        }
    }
}

fn fill_global_settings(node: &BinNode, doc: &mut FbxDoc) {
    for ch in &node.children {
        if ch.name == "Properties70" || ch.name == "Properties60" {
            for p in &ch.children {
                if p.name != "P" {
                    continue;
                }
                let key = prop_string(p.props.get(0)).unwrap_or_default();
                // Exact match: "OriginalUnitScaleFactor" is the unit of the authoring app,
                // not the unit the stored numbers are in.
                let nums = prop_trailing_f32(p);
                if key == "UnitScaleFactor" {
                    if let Some(&v) = nums.last() {
                        doc.unit_to_meters = unit_factor_to_meters(v);
                    }
                } else if key == "UpAxis" {
                    if let Some(&v) = nums.last() {
                        doc.up_axis = v as i32;
                    }
                } else if key == "UpAxisSign" {
                    if let Some(&v) = nums.last() {
                        doc.up_sign = if v < 0.0 { -1 } else { 1 };
                    }
                }
            }
        }
    }
}

fn fill_material_from_children(node: &BinNode, mat: &mut Mat) {
    for ch in &node.children {
        if ch.name == "Properties70" || ch.name == "Properties60" {
            for p in &ch.children {
                if p.name != "P" {
                    continue;
                }
                let key = prop_string(p.props.get(0)).unwrap_or_default();
                let nums = prop_trailing_f32(p);
                let rgb = |n: &Vec<f32>| {
                    if n.len() >= 3 {
                        Some([n[n.len() - 3], n[n.len() - 2], n[n.len() - 1]])
                    } else {
                        None
                    }
                };
                if key == "DiffuseColor"
                    || key == "Diffuse"
                    || key.to_ascii_lowercase().contains("base_color")
                    || key.to_ascii_lowercase().contains("basecolor")
                {
                    if let Some(c) = rgb(&nums) {
                        mat.diffuse = c;
                    }
                } else if key == "EmissiveColor" || key == "Emissive" {
                    if let Some(c) = rgb(&nums) {
                        mat.emissive = c;
                    }
                } else if key == "Opacity" {
                    if let Some(&o) = nums.last() {
                        mat.opacity = o.clamp(0.0, 1.0);
                    }
                }
            }
        }
    }
}

fn parse_bin_connection(node: &BinNode, doc: &mut FbxDoc) {
    // C: "OO", child, parent      C: "OP", child, parent, "PropertyName"
    let ty = prop_string(node.props.get(0)).unwrap_or_else(|| "OO".into());
    let (Some(child), Some(parent)) = (
        prop_i64(node.props.get(1)),
        prop_i64(node.props.get(2)),
    ) else {
        return;
    };
    doc.connections.push(Conn {
        ty,
        child,
        parent,
        prop: prop_string(node.props.get(3)).unwrap_or_default(),
    });
}

fn prop_ktime(p: &BinNode) -> Option<i64> {
    // Properties70 P entries: name, type, ..., value
    for prop in &p.props {
        match prop {
            BinProp::I64(v) => return Some(*v),
            BinProp::I32(v) => return Some(*v as i64),
            BinProp::F64(v) => return Some((*v) as i64),
            BinProp::F32(v) => return Some((*v) as i64),
            _ => {}
        }
    }
    None
}

fn prop_i64(p: Option<&BinProp>) -> Option<i64> {
    match p? {
        BinProp::I64(v) => Some(*v),
        BinProp::I32(v) => Some(*v as i64),
        BinProp::I16(v) => Some(*v as i64),
        _ => None,
    }
}

fn prop_string(p: Option<&BinProp>) -> Option<String> {
    match p? {
        BinProp::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn prop_trailing_f32(node: &BinNode) -> Vec<f32> {
    let mut v = Vec::new();
    for p in &node.props {
        match p {
            BinProp::F32(x) => v.push(*x),
            BinProp::F64(x) => v.push(*x as f32),
            BinProp::I32(x) => v.push(*x as f32),
            BinProp::I64(x) => v.push(*x as f32),
            _ => {}
        }
    }
    v
}

// ===========================================================================
// Binary FBX 7.4 writer (local, no external FBX libraries)
// Version 7400 uses 32-bit offsets.
// ===========================================================================

pub fn build_binary_fbx(
    scene: &Scene,
    only: Option<&BTreeSet<u64>>,
    tex_rel: &HashMap<usize, String>,
) -> Result<Vec<u8>> {
    let mut meshes: Vec<(String, MeshData, [f32; 4], usize)> = Vec::new(); // name, mesh, diffuse, mat_idx
    // (name, color, optional diffuse texture index into scene.textures)
    let mut mat_names: Vec<(String, [f32; 4], Option<usize>)> = scene
        .materials
        .iter()
        .map(|m| (m.name.clone(), m.base_color, m.base_color_tex))
        .collect();
    if mat_names.is_empty() {
        mat_names.push(("Default".into(), [0.8, 0.8, 0.8, 1.0], None));
    }

    // Armature bones exported as LimbNode models (hierarchy only — no skin weights yet).
    let mut limb_models: Vec<(String, [f32; 3], [f32; 4], Option<usize>)> = Vec::new(); // name, head, rest_orient, parent_limb_idx
    for obj in &scene.objects {
        if let Some(sel) = only {
            if !sel.contains(&obj.id) {
                continue;
            }
        }
        if let ObjectKind::Armature(arm) = &obj.kind {
            let base = limb_models.len();
            for (bi, bone) in arm.bones.iter().enumerate() {
                let parent = bone.parent.map(|p| base + p as usize);
                limb_models.push((
                    bone.name.clone(),
                    bone.head,
                    bone.rest_orientation.unwrap_or([0.0, 0.0, 0.0, 1.0]),
                    parent,
                ));
                let _ = bi;
            }
        }
    }

    for obj in &scene.objects {
        if let Some(sel) = only {
            if !sel.contains(&obj.id) {
                continue;
            }
        }
        match &obj.kind {
            ObjectKind::Mesh => {}
            _ => continue, // non-mesh helpers (surface, volume, armature, etc.)
        }
        let world = Mat4::from_scale_rotation_translation(obj.scale, obj.rotation, obj.translation);
        let baked = obj.mesh.transformed(world);
        if baked.vertices.is_empty() || baked.indices.is_empty() {
            continue;
        }
        let mi = obj
            .material
            .unwrap_or(0)
            .min(mat_names.len().saturating_sub(1));
        let diff = mat_names
            .get(mi)
            .map(|(_, c, _)| *c)
            .unwrap_or([0.8, 0.8, 0.8, 1.0]);
        meshes.push((obj.name.clone(), baked, diff, mi));
    }
    if meshes.is_empty() && limb_models.is_empty() {
        bail!("nothing to export as FBX (no mesh or armature objects)");
    }

    // Collect unique materials referenced
    let mut used_mats: Vec<(String, [f32; 4], Option<usize>)> = Vec::new();
    let mut mat_remap: Vec<usize> = vec![0; mat_names.len()];
    for (_, _, _, mi) in &meshes {
        if used_mats.iter().all(|(n, _, _)| n != &mat_names[*mi].0) {
            mat_remap[*mi] = used_mats.len();
            used_mats.push(mat_names[*mi].clone());
        } else {
            mat_remap[*mi] = used_mats
                .iter()
                .position(|(n, _, _)| n == &mat_names[*mi].0)
                .unwrap_or(0);
        }
    }
    if used_mats.is_empty() {
        used_mats.push(("Default".into(), [0.8, 0.8, 0.8, 1.0], None));
    }

    let mut next_id: i64 = 100000;
    let mut mat_ids: Vec<i64> = Vec::new();
    for _ in &used_mats {
        mat_ids.push(next_id);
        next_id += 1;
    }
    let mut geom_ids: Vec<i64> = Vec::new();
    let mut model_ids: Vec<i64> = Vec::new();
    for _ in &meshes {
        geom_ids.push(next_id);
        next_id += 1;
        model_ids.push(next_id);
        next_id += 1;
    }

    // Build node tree
    let header_ext = bin_node(
        "FBXHeaderExtension",
        vec![],
        vec![
            bin_node("FBXHeaderVersion", vec![BinWProp::I32(1003)], vec![]),
            bin_node("FBXVersion", vec![BinWProp::I32(7400)], vec![]),
        ],
    );

    let global = bin_node(
        "GlobalSettings",
        vec![],
        vec![
            bin_node("Version", vec![BinWProp::I32(1000)], vec![]),
            bin_node(
                "Properties70",
                vec![],
                vec![
                    prop70_i("UpAxis", 1),
                    prop70_i("UpAxisSign", 1),
                    prop70_i("FrontAxis", 2),
                    prop70_i("FrontAxisSign", 1),
                    prop70_i("CoordAxis", 0),
                    prop70_i("CoordAxisSign", 1),
                    prop70_d("UnitScaleFactor", 100.0),
                    prop70_d("OriginalUnitScaleFactor", 100.0),
                ],
            ),
        ],
    );

    // Texture / Video ids for materials that have a diffuse map on disk.
    let mut tex_ids: Vec<Option<(i64, i64)>> = vec![None; used_mats.len()]; // (texture_id, video_id)
    let mut next_tex_id = next_id;
    for (i, (_, _, tex_opt)) in used_mats.iter().enumerate() {
        if let Some(ti) = tex_opt {
            if tex_rel.contains_key(ti) {
                let vid = next_tex_id;
                next_tex_id += 1;
                let tid = next_tex_id;
                next_tex_id += 1;
                tex_ids[i] = Some((tid, vid));
            }
        }
    }
    next_id = next_tex_id;

    // LimbNode model ids for exported bones.
    let mut limb_ids: Vec<i64> = Vec::with_capacity(limb_models.len());
    for _ in &limb_models {
        limb_ids.push(next_id);
        next_id += 1;
    }

    let model_count = (meshes.len() + limb_models.len()) as i32;
    let tex_count = tex_ids.iter().filter(|t| t.is_some()).count() as i32;
    let definitions = bin_node(
        "Definitions",
        vec![],
        vec![
            bin_node("Version", vec![BinWProp::I32(100)], vec![]),
            bin_node(
                "Count",
                vec![BinWProp::I32(3 + if tex_count > 0 { 2 } else { 0 })],
                vec![],
            ),
            object_type_def("Geometry", meshes.len() as i32),
            object_type_def("Model", model_count),
            object_type_def("Material", used_mats.len() as i32),
            object_type_def("Texture", tex_count),
            object_type_def("Video", tex_count),
        ],
    );

    let mut object_children: Vec<BinWNode> = Vec::new();
    for (i, (name, color, _)) in used_mats.iter().enumerate() {
        object_children.push(material_node(mat_ids[i], name, *color));
    }
    for (i, (name, mesh, _, mi)) in meshes.iter().enumerate() {
        object_children.push(geometry_node(geom_ids[i], name, mesh));
        object_children.push(model_node(model_ids[i], name));
        let _ = mi;
    }
    // Diffuse textures as Texture + Video pairs with RelativeFilename.
    for (i, (_, _, tex_opt)) in used_mats.iter().enumerate() {
        let Some(ti) = tex_opt else { continue };
        let Some(fname) = tex_rel.get(ti) else { continue };
        let Some((tid, vid)) = tex_ids[i] else { continue };
        object_children.push(bin_node(
            "Video",
            vec![
                BinWProp::I64(vid),
                BinWProp::String(format!("Video::{}", sanitize(fname))),
                BinWProp::String("Clip".into()),
            ],
            vec![
                bin_node("Type", vec![BinWProp::String("Clip".into())], vec![]),
                bin_node(
                    "Properties70",
                    vec![],
                    vec![bin_node(
                        "P",
                        vec![
                            BinWProp::String("Path".into()),
                            BinWProp::String("KString".into()),
                            BinWProp::String("XRefUrl".into()),
                            BinWProp::String("".into()),
                            BinWProp::String(fname.clone()),
                        ],
                        vec![],
                    )],
                ),
                bin_node(
                    "UseMipMap",
                    vec![BinWProp::I32(0)],
                    vec![],
                ),
                bin_node(
                    "Filename",
                    vec![BinWProp::String(fname.clone())],
                    vec![],
                ),
                bin_node(
                    "RelativeFilename",
                    vec![BinWProp::String(fname.clone())],
                    vec![],
                ),
            ],
        ));
        object_children.push(bin_node(
            "Texture",
            vec![
                BinWProp::I64(tid),
                BinWProp::String(format!("Texture::{}", sanitize(fname))),
                BinWProp::String("".into()),
            ],
            vec![
                bin_node("Type", vec![BinWProp::String("TextureVideoClip".into())], vec![]),
                bin_node("Version", vec![BinWProp::I32(202)], vec![]),
                bin_node(
                    "TextureName",
                    vec![BinWProp::String(format!("Texture::{}", sanitize(fname)))],
                    vec![],
                ),
                bin_node(
                    "Media",
                    vec![BinWProp::String(format!("Video::{}", sanitize(fname)))],
                    vec![],
                ),
                bin_node(
                    "Filename",
                    vec![BinWProp::String(fname.clone())],
                    vec![],
                ),
                bin_node(
                    "RelativeFilename",
                    vec![BinWProp::String(fname.clone())],
                    vec![],
                ),
            ],
        ));
    }
    // Bones as LimbNode models.
    for (i, (name, head, _orient, _parent)) in limb_models.iter().enumerate() {
        object_children.push(bin_node(
            "Model",
            vec![
                BinWProp::I64(limb_ids[i]),
                BinWProp::String(format!("Model::{}", sanitize(name))),
                BinWProp::String("LimbNode".into()),
            ],
            vec![
                bin_node("Version", vec![BinWProp::I32(232)], vec![]),
                bin_node(
                    "Properties70",
                    vec![],
                    vec![
                        prop70_i("InheritType", 1),
                        bin_node(
                            "P",
                            vec![
                                BinWProp::String("Lcl Translation".into()),
                                BinWProp::String("Lcl Translation".into()),
                                BinWProp::String("".into()),
                                BinWProp::String("A".into()),
                                BinWProp::F64(head[0] as f64),
                                BinWProp::F64(head[1] as f64),
                                BinWProp::F64(head[2] as f64),
                            ],
                            vec![],
                        ),
                    ],
                ),
            ],
        ));
    }
    let objects = bin_node("Objects", vec![], object_children);

    let mut conn_children: Vec<BinWNode> = Vec::new();
    for i in 0..meshes.len() {
        // Geometry -> Model
        conn_children.push(connection_oo(geom_ids[i], model_ids[i]));
        // Model -> root (0)
        conn_children.push(connection_oo(model_ids[i], 0));
        let mi = meshes[i].3;
        let mid = mat_ids[mat_remap
            .get(mi)
            .copied()
            .unwrap_or(0)
            .min(mat_ids.len() - 1)];
        conn_children.push(connection_oo(mid, model_ids[i]));
        // Texture -> Material DiffuseColor, Video -> Texture
        let remap_i = mat_remap.get(mi).copied().unwrap_or(0).min(tex_ids.len().saturating_sub(1));
        if let Some((tid, vid)) = tex_ids.get(remap_i).copied().flatten() {
            conn_children.push(connection_op(tid, mid, "DiffuseColor"));
            conn_children.push(connection_oo(vid, tid));
        }
    }
    // Limb hierarchy
    for (i, (_, _, _, parent)) in limb_models.iter().enumerate() {
        let parent_id = parent.map(|p| limb_ids[p]).unwrap_or(0);
        conn_children.push(connection_oo(limb_ids[i], parent_id));
    }
    let connections = bin_node("Connections", vec![], conn_children);

    // Write file
    let mut out = Vec::new();
    // Magic header 23 bytes + version
    out.extend_from_slice(b"Kaydara FBX Binary  \x00");
    out.extend_from_slice(&[0x1A, 0x00]);
    out.extend_from_slice(&7400u32.to_le_bytes());

    write_node(&mut out, &header_ext);
    write_node(&mut out, &global);
    write_node(&mut out, &definitions);
    write_node(&mut out, &objects);
    write_node(&mut out, &connections);
    // Final null record (13 bytes for v7400)
    out.extend_from_slice(&[0u8; 13]);
    Ok(out)
}

#[derive(Clone)]
enum BinWProp {
    Bool(bool),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    String(String),
    I32Arr(Vec<i32>),
    F64Arr(Vec<f64>),
}

struct BinWNode {
    name: String,
    props: Vec<BinWProp>,
    children: Vec<BinWNode>,
}

fn bin_node(name: &str, props: Vec<BinWProp>, children: Vec<BinWNode>) -> BinWNode {
    BinWNode {
        name: name.to_string(),
        props,
        children,
    }
}

fn object_type_def(ty: &str, count: i32) -> BinWNode {
    bin_node(
        "ObjectType",
        vec![BinWProp::String(ty.into())],
        vec![bin_node("Count", vec![BinWProp::I32(count)], vec![])],
    )
}

fn prop70_i(name: &str, v: i32) -> BinWNode {
    bin_node(
        "P",
        vec![
            BinWProp::String(name.into()),
            BinWProp::String("int".into()),
            BinWProp::String("Integer".into()),
            BinWProp::String("".into()),
            BinWProp::I32(v),
        ],
        vec![],
    )
}

fn prop70_d(name: &str, v: f64) -> BinWNode {
    bin_node(
        "P",
        vec![
            BinWProp::String(name.into()),
            BinWProp::String("double".into()),
            BinWProp::String("Number".into()),
            BinWProp::String("".into()),
            BinWProp::F64(v),
        ],
        vec![],
    )
}

fn prop70_color(name: &str, r: f64, g: f64, b: f64) -> BinWNode {
    bin_node(
        "P",
        vec![
            BinWProp::String(name.into()),
            BinWProp::String("Color".into()),
            BinWProp::String("".into()),
            BinWProp::String("A".into()),
            BinWProp::F64(r),
            BinWProp::F64(g),
            BinWProp::F64(b),
        ],
        vec![],
    )
}

fn material_node(id: i64, name: &str, color: [f32; 4]) -> BinWNode {
    bin_node(
        "Material",
        vec![
            BinWProp::I64(id),
            BinWProp::String(format!("Material::{}", sanitize(name))),
            BinWProp::String("".into()),
        ],
        vec![
            bin_node("Version", vec![BinWProp::I32(102)], vec![]),
            bin_node(
                "ShadingModel",
                vec![BinWProp::String("Phong".into())],
                vec![],
            ),
            bin_node("MultiLayer", vec![BinWProp::I32(0)], vec![]),
            bin_node(
                "Properties70",
                vec![],
                vec![prop70_color(
                    "DiffuseColor",
                    color[0] as f64,
                    color[1] as f64,
                    color[2] as f64,
                )],
            ),
        ],
    )
}

fn geometry_node(id: i64, name: &str, mesh: &MeshData) -> BinWNode {
    let mut verts: Vec<f64> = Vec::with_capacity(mesh.vertices.len() * 3);
    let mut norms: Vec<f64> = Vec::with_capacity(mesh.vertices.len() * 3);
    let mut uvs: Vec<f64> = Vec::with_capacity(mesh.vertices.len() * 2);
    for v in &mesh.vertices {
        verts.extend_from_slice(&[
            v.position[0] as f64,
            v.position[1] as f64,
            v.position[2] as f64,
        ]);
        norms.extend_from_slice(&[v.normal[0] as f64, v.normal[1] as f64, v.normal[2] as f64]);
        // FBX UV V is typically top-origin; BlendCraft stores bottom-origin like glTF.
        uvs.extend_from_slice(&[v.uv[0] as f64, 1.0 - v.uv[1] as f64]);
    }
    let mut pvi: Vec<i32> = Vec::with_capacity(mesh.indices.len());
    for tri in mesh.indices.chunks_exact(3) {
        pvi.push(tri[0] as i32);
        pvi.push(tri[1] as i32);
        pvi.push(-(tri[2] as i32) - 1);
    }

    bin_node(
        "Geometry",
        vec![
            BinWProp::I64(id),
            BinWProp::String(format!("Geometry::{}", sanitize(name))),
            BinWProp::String("Mesh".into()),
        ],
        vec![
            bin_node("Vertices", vec![BinWProp::F64Arr(verts)], vec![]),
            bin_node("PolygonVertexIndex", vec![BinWProp::I32Arr(pvi)], vec![]),
            bin_node("GeometryVersion", vec![BinWProp::I32(124)], vec![]),
            bin_node(
                "LayerElementNormal",
                vec![BinWProp::I32(0)],
                vec![
                    bin_node("Version", vec![BinWProp::I32(101)], vec![]),
                    bin_node("Name", vec![BinWProp::String("".into())], vec![]),
                    bin_node(
                        "MappingInformationType",
                        vec![BinWProp::String("ByVertice".into())],
                        vec![],
                    ),
                    bin_node(
                        "ReferenceInformationType",
                        vec![BinWProp::String("Direct".into())],
                        vec![],
                    ),
                    bin_node("Normals", vec![BinWProp::F64Arr(norms)], vec![]),
                ],
            ),
            bin_node(
                "LayerElementUV",
                vec![BinWProp::I32(0)],
                vec![
                    bin_node("Version", vec![BinWProp::I32(101)], vec![]),
                    bin_node("Name", vec![BinWProp::String("UVMap".into())], vec![]),
                    bin_node(
                        "MappingInformationType",
                        vec![BinWProp::String("ByVertice".into())],
                        vec![],
                    ),
                    bin_node(
                        "ReferenceInformationType",
                        vec![BinWProp::String("Direct".into())],
                        vec![],
                    ),
                    bin_node("UV", vec![BinWProp::F64Arr(uvs)], vec![]),
                ],
            ),
            bin_node(
                "Layer",
                vec![BinWProp::I32(0)],
                vec![
                    bin_node("Version", vec![BinWProp::I32(100)], vec![]),
                    bin_node(
                        "LayerElement",
                        vec![],
                        vec![
                            bin_node(
                                "Type",
                                vec![BinWProp::String("LayerElementNormal".into())],
                                vec![],
                            ),
                            bin_node("TypedIndex", vec![BinWProp::I32(0)], vec![]),
                        ],
                    ),
                    bin_node(
                        "LayerElement",
                        vec![],
                        vec![
                            bin_node(
                                "Type",
                                vec![BinWProp::String("LayerElementUV".into())],
                                vec![],
                            ),
                            bin_node("TypedIndex", vec![BinWProp::I32(0)], vec![]),
                        ],
                    ),
                ],
            ),
        ],
    )
}

fn model_node(id: i64, name: &str) -> BinWNode {
    bin_node(
        "Model",
        vec![
            BinWProp::I64(id),
            BinWProp::String(format!("Model::{}", sanitize(name))),
            BinWProp::String("Mesh".into()),
        ],
        vec![
            bin_node("Version", vec![BinWProp::I32(232)], vec![]),
            bin_node(
                "Properties70",
                vec![],
                vec![
                    prop70_i("InheritType", 1),
                    prop70_i("DefaultAttributeIndex", 0),
                    // Identity Lcl — geometry already baked
                    bin_node(
                        "P",
                        vec![
                            BinWProp::String("Lcl Translation".into()),
                            BinWProp::String("Lcl Translation".into()),
                            BinWProp::String("".into()),
                            BinWProp::String("A".into()),
                            BinWProp::F64(0.0),
                            BinWProp::F64(0.0),
                            BinWProp::F64(0.0),
                        ],
                        vec![],
                    ),
                    bin_node(
                        "P",
                        vec![
                            BinWProp::String("Lcl Rotation".into()),
                            BinWProp::String("Lcl Rotation".into()),
                            BinWProp::String("".into()),
                            BinWProp::String("A".into()),
                            BinWProp::F64(0.0),
                            BinWProp::F64(0.0),
                            BinWProp::F64(0.0),
                        ],
                        vec![],
                    ),
                    bin_node(
                        "P",
                        vec![
                            BinWProp::String("Lcl Scaling".into()),
                            BinWProp::String("Lcl Scaling".into()),
                            BinWProp::String("".into()),
                            BinWProp::String("A".into()),
                            BinWProp::F64(1.0),
                            BinWProp::F64(1.0),
                            BinWProp::F64(1.0),
                        ],
                        vec![],
                    ),
                ],
            ),
            bin_node("Shading", vec![BinWProp::Bool(true)], vec![]),
            bin_node(
                "Culling",
                vec![BinWProp::String("CullingOff".into())],
                vec![],
            ),
        ],
    )
}

fn connection_oo(child: i64, parent: i64) -> BinWNode {
    bin_node(
        "C",
        vec![
            BinWProp::String("OO".into()),
            BinWProp::I64(child),
            BinWProp::I64(parent),
        ],
        vec![],
    )
}

fn connection_op(child: i64, parent: i64, prop: &str) -> BinWNode {
    bin_node(
        "C",
        vec![
            BinWProp::String("OP".into()),
            BinWProp::I64(child),
            BinWProp::I64(parent),
            BinWProp::String(prop.into()),
        ],
        vec![],
    )
}

fn write_node(out: &mut Vec<u8>, node: &BinWNode) {
    let start = out.len();
    // Placeholders: EndOffset, NumProperties, PropertyListLen (3 x u32 for v7400)
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(node.props.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // prop list len placeholder

    let name_bytes = node.name.as_bytes();
    out.push(name_bytes.len() as u8);
    out.extend_from_slice(name_bytes);

    let prop_start = out.len();
    for p in &node.props {
        write_prop(out, p);
    }
    let prop_len = (out.len() - prop_start) as u32;
    // patch PropertyListLen at start+8
    out[start + 8..start + 12].copy_from_slice(&prop_len.to_le_bytes());

    for ch in &node.children {
        write_node(out, ch);
    }
    // null terminator for children list
    out.extend_from_slice(&[0u8; 13]);

    let end = out.len() as u32;
    out[start..start + 4].copy_from_slice(&end.to_le_bytes());
}

fn write_prop(out: &mut Vec<u8>, p: &BinWProp) {
    match p {
        BinWProp::Bool(v) => {
            out.push(b'C');
            out.push(if *v { 1 } else { 0 });
        }
        BinWProp::I32(v) => {
            out.push(b'I');
            out.extend_from_slice(&v.to_le_bytes());
        }
        BinWProp::I64(v) => {
            out.push(b'L');
            out.extend_from_slice(&v.to_le_bytes());
        }
        BinWProp::F32(v) => {
            out.push(b'F');
            out.extend_from_slice(&v.to_le_bytes());
        }
        BinWProp::F64(v) => {
            out.push(b'D');
            out.extend_from_slice(&v.to_le_bytes());
        }
        BinWProp::String(s) => {
            out.push(b'S');
            let b = s.as_bytes();
            out.extend_from_slice(&(b.len() as u32).to_le_bytes());
            out.extend_from_slice(b);
        }
        BinWProp::I32Arr(arr) => {
            out.push(b'i');
            write_array_i32(out, arr);
        }
        BinWProp::F64Arr(arr) => {
            out.push(b'd');
            write_array_f64(out, arr);
        }
    }
}

fn write_array_i32(out: &mut Vec<u8>, arr: &[i32]) {
    out.extend_from_slice(&(arr.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // encoding raw
    let data_len = (arr.len() * 4) as u32;
    out.extend_from_slice(&data_len.to_le_bytes());
    for v in arr {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn write_array_f64(out: &mut Vec<u8>, arr: &[f64]) {
    out.extend_from_slice(&(arr.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // encoding raw
    let data_len = (arr.len() * 8) as u32;
    out.extend_from_slice(&data_len.to_le_bytes());
    for v in arr {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::primitives;

    #[test]
    fn ascii_roundtrip_cube() {
        let mut scene = Scene::default();
        scene.materials.push(Material {
            name: "Red".into(),
            base_color: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        });
        let mesh = primitives::cube(1.0); // ~1m cube centered at origin
        let mut obj = Object::new("Cube", mesh);
        obj.material = Some(0);
        scene.objects.push(obj);

        let text = build_ascii_fbx(&scene, None).expect("export");
        assert!(text.contains("Vertices:"));
        assert!(text.contains("UnitScaleFactor"));
        assert!(text.contains("PolygonVertexIndex:"));

        let loaded = load_fbx_str(&text).expect("import");
        assert!(!loaded.objects.is_empty());
        let m = &loaded.objects[0].mesh;
        assert!(m.vertices.len() >= 8);
        assert!(m.indices.len() >= 36);
        // Size must stay ~1m, not 100x or 0.01x
        let mut max_abs = 0.0f32;
        for v in &m.vertices {
            for c in v.position {
                max_abs = max_abs.max(c.abs());
            }
        }
        assert!(
            max_abs > 0.4 && max_abs < 2.0,
            "cube extent {max_abs} is wrong scale (expected ~0.5)"
        );
    }

    #[test]
    fn binary_roundtrip_cube() {
        let mut scene = Scene::default();
        scene.materials.push(Material {
            name: "Red".into(),
            base_color: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        });
        let mesh = primitives::cube(1.0);
        let mut obj = Object::new("Cube", mesh);
        obj.material = Some(0);
        scene.objects.push(obj);

        let bytes = build_binary_fbx(&scene, None, &HashMap::new()).expect("export binary");
        assert!(
            bytes.starts_with(b"Kaydara FBX Binary"),
            "missing binary magic"
        );
        assert!(bytes.len() > 100);

        let loaded = load_fbx_bytes(&bytes).expect("import binary");
        assert!(!loaded.objects.is_empty());
        let m = &loaded.objects[0].mesh;
        assert!(m.vertices.len() >= 8);
        assert!(m.indices.len() >= 36);
        let mut max_abs = 0.0f32;
        for v in &m.vertices {
            for c in v.position {
                max_abs = max_abs.max(c.abs());
            }
        }
        assert!(
            max_abs > 0.4 && max_abs < 2.0,
            "binary cube extent {max_abs} is wrong scale"
        );
    }

    #[test]
    fn unit_scale_cm_file() {
        // Minimal ASCII FBX in centimeters (UnitScaleFactor=1), 100cm cube → 1m
        let text = r#"; FBX 7.4.0
FBXHeaderExtension:  {
    FBXVersion: 7400
}
GlobalSettings:  {
    Properties70:  {
        P: "UnitScaleFactor", "double", "Number", "",1
    }
}
Objects:  {
    Geometry: 1, "Geometry::Box", "Mesh" {
        Vertices: *24 {
            a: -50,-50,-50, 50,-50,-50, 50,50,-50, -50,50,-50
        }
        PolygonVertexIndex: *6 {
            a: 0,1,-3
        }
    }
    Model: 2, "Model::Box", "Mesh" {
    }
}
Connections:  {
    C: "OO",1,2
    C: "OO",2,0
}
"#;
        let loaded = load_fbx_str(text).expect("import cm");
        let m = &loaded.objects[0].mesh;
        let mut max_abs = 0.0f32;
        for v in &m.vertices {
            for c in v.position {
                max_abs = max_abs.max(c.abs());
            }
        }
        // 50 cm → 0.5 m
        assert!((max_abs - 0.5).abs() < 0.05, "expected 0.5m, got {max_abs}");
    }

    #[test]
    fn mesh_parented_to_bone_keeps_hierarchy() {
        // 100 file units per metre -> factor 1.0
        let text = r#"; FBX 7.4.0
GlobalSettings:  {
    Properties70:  {
        P: "UnitScaleFactor", "double", "Number", "",100
    }
}
Objects:  {
    Geometry: 1, "Geometry::Tri", "Mesh" {
        Vertices: *9 {
            a: 0,0,0, 1,0,0, 0,1,0
        }
        PolygonVertexIndex: *3 {
            a: 0,1,-3
        }
    }
    Model: 2, "Model::Body", "Mesh" {
    }
    Model: 3, "Model::Bone", "LimbNode" {
        Properties70:  {
            P: "Lcl Translation", "Lcl Translation", "", "A",5,0,0
        }
    }
}
Connections:  {
    C: "OO",1,2
    C: "OO",2,3
    C: "OO",3,0
}
"#;
        let loaded = load_fbx_str(text).expect("import");
        assert_eq!(loaded.objects.len(), 1);
        let x = loaded.objects[0].translation.x;
        assert!((x - 5.0).abs() < 0.01, "bone offset lost, x = {x}");
    }

    #[test]
    fn uvs_follow_polygon_vertex_mapping() {
        let text = r#"; FBX 7.4.0
Objects:  {
    Geometry: 1, "Geometry::Tri", "Mesh" {
        Vertices: *9 {
            a: 0,0,0, 1,0,0, 0,1,0
        }
        PolygonVertexIndex: *3 {
            a: 0,1,-3
        }
        LayerElementUV: 0 {
            MappingInformationType: "ByPolygonVertex"
            ReferenceInformationType: "IndexToDirect"
            UV: *6 {
                a: 0,0, 1,0, 0,1
            }
            UVIndex: *3 {
                a: 2,1,0
            }
        }
    }
    Model: 2, "Model::Tri", "Mesh" {
    }
}
Connections:  {
    C: "OO",1,2
}
"#;
        let loaded = load_fbx_str(text).expect("import");
        let m = &loaded.objects[0].mesh;
        assert_eq!(m.vertices.len(), 3);
        // corner k uses UV[UVIndex[k]], v is flipped
        assert_eq!(m.vertices[0].uv, [0.0, 0.0]);
        assert_eq!(m.vertices[1].uv, [1.0, 1.0]);
        assert_eq!(m.vertices[2].uv, [0.0, 1.0]);
    }

    #[test]
    fn embedded_texture_reaches_material() {
        let text = format!(
            r#"; FBX 7.4.0
Objects:  {{
    Geometry: 1, "Geometry::Tri", "Mesh" {{
        Vertices: *9 {{
            a: 0,0,0, 1,0,0, 0,1,0
        }}
        PolygonVertexIndex: *3 {{
            a: 0,1,-3
        }}
    }}
    Model: 2, "Model::Tri", "Mesh" {{
    }}
    Material: 20, "Material::mat", "" {{
        Properties70:  {{
            P: "DiffuseColor", "Color", "", "A",0.5,0.5,0.5
        }}
    }}
    Texture: 10, "Texture::tex", "" {{
        FileName: "C:\art\tex.png"
        RelativeFilename: "tex.png"
    }}
    Video: 11, "Video::tex", "Clip" {{
        Filename: "C:\art\tex.png"
        Content: "{}"
    }}
}}
Connections:  {{
    C: "OO",1,2
    C: "OO",20,2
    C: "OO",11,10
    C: "OP",10,20, "DiffuseColor"
}}
"#,
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
        );
        let loaded = load_fbx_str(&text).expect("import");
        assert_eq!(loaded.textures.len(), 1);
        assert_eq!((loaded.textures[0].width, loaded.textures[0].height), (1, 1));
        let mat = &loaded.materials[loaded.objects[0].material.expect("material")];
        assert_eq!(mat.base_color_tex, Some(0));
    }
}
