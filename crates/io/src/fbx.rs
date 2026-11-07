//! Local ASCII FBX 7.x import/export — no external FBX crates or online APIs.
//!
//! Supports a practical subset:
//! - Import: Vertices, PolygonVertexIndex, Normals, UV, Materials (DiffuseColor),
//!   Model nodes, Geometry, Connections (OO)
//! - Export: same subset as static triangle meshes + Phong materials
use anyhow::{bail, Context, Result};
use forge_core::{Material, MeshData, Object, ObjectKind, Scene, Vertex};
use glam::{Mat4, Quat, Vec3};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
pub fn load_fbx(path: &Path) -> Result<Scene> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading {}", path.display()))?;
    load_fbx_bytes(&bytes)
}

#[cfg(target_arch = "wasm32")]
pub fn load_fbx(_path: &Path) -> Result<Scene> {
    bail!("FBX load from path is not available on WebAssembly")
}

/// Parse ASCII FBX text into a BlendCraft scene.
pub fn load_fbx_str(text: &str) -> Result<Scene> {
    let doc = parse_ascii(text)?;
    convert_doc(&doc)
}

/// Load FBX from bytes — auto-detects binary vs ASCII. Fully local, no external crates.
pub fn load_fbx_bytes(bytes: &[u8]) -> Result<Scene> {
    if is_binary_fbx(bytes) {
        let doc = parse_binary(bytes)?;
        return convert_doc(&doc);
    }
    let text = std::str::from_utf8(bytes).context("FBX is not valid UTF-8 (and not a binary FBX header)")?;
    load_fbx_str(text)
}

fn is_binary_fbx(bytes: &[u8]) -> bool {
    bytes.len() >= 27 && bytes.starts_with(b"Kaydara FBX Binary")
}

#[cfg(not(target_arch = "wasm32"))]
pub fn export_fbx(scene: &Scene, path: &Path, only: Option<&BTreeSet<u64>>) -> Result<()> {
    // Binary FBX 7.4 — Blender and most DCCs import this; ASCII is often rejected.
    let bytes = build_binary_fbx(scene, only)?;
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

#[cfg(target_arch = "wasm32")]
pub fn export_fbx(_scene: &Scene, _path: &Path, _only: Option<&BTreeSet<u64>>) -> Result<()> {
    bail!("FBX export is not available on WebAssembly")
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
    /// connections: (child_id, parent_id) for OO links
    connections: Vec<(i64, i64)>,
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
            connections: Vec::new(),
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
            parent: None,
        }
    }
}

#[derive(Debug, Default)]
struct Mat {
    name: String,
    diffuse: [f32; 3],
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
            let kind = name_kind(&name);
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
            let mut m = Mat {
                name: strip_name(&name),
                diffuse: [0.8, 0.8, 0.8],
            };
            parse_material_body(&body, &mut m);
            doc.materials.insert(id, m);
            continue;
        }
        if line.starts_with("Connections:") {
            let (_, _, body, next) = read_block(&lines, i)?;
            i = next;
            for bl in body.lines() {
                let bl = bl.trim();
                // C: "OO",child,parent
                if bl.starts_with("C:") {
                    if let Some((a, b)) = parse_connection(bl) {
                        doc.connections.push((a, b));
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

fn name_kind(full: &str) -> String {
    // header has third quoted field sometimes; fallback Mesh
    if full.contains("Null") {
        "Null".into()
    } else {
        "Mesh".into()
    }
}

fn parse_connection(line: &str) -> Option<(i64, i64)> {
    // C: "OO",123,456
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 3 {
        return None;
    }
    let a = parts[1].trim().parse().ok()?;
    let b = parts[2].trim().split_whitespace().next()?.parse().ok()?;
    Some((a, b))
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
    // Normals often under LayerElementNormal → Normals:
    if let Some(v) = extract_array_f32(body, "Normals:") {
        for chunk in v.chunks_exact(3) {
            g.normals.push([chunk[0], chunk[1], chunk[2]]);
        }
    }
    if let Some(v) = extract_array_f32(body, "UV:") {
        for chunk in v.chunks_exact(2) {
            g.uvs.push([chunk[0], chunk[1]]);
        }
    }
    if let Some(v) = extract_array_i32(body, "UVIndex:") {
        g.uv_indices = v;
    }
}

fn parse_model_body(body: &str, m: &mut Model) {
    for key in [
        "Lcl Translation", "Lcl Rotation", "Lcl Scaling",
        "GeometricTranslation", "GeometricRotation", "GeometricScaling",
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
}

fn parse_material_body(body: &str, m: &mut Mat) {
    if let Some(v) = extract_property_vec3(body, "DiffuseColor") {
        m.diffuse = v;
    } else if let Some(v) = extract_property_vec3(body, "Diffuse") {
        m.diffuse = v;
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
        if t.contains(':') && !t.contains(',') && !t.chars().next().map(|c| c.is_ascii_digit() || c == '-' || c == '.').unwrap_or(false) {
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
    for key in ["UnitScaleFactor", "OriginalUnitScaleFactor"] {
        if let Some(v) = extract_property_scalar(body, key) {
            doc.unit_to_meters = unit_factor_to_meters(v);
            break;
        }
    }
}

fn extract_property_scalar(body: &str, key: &str) -> Option<f32> {
    for line in body.lines() {
        let t = line.trim();
        if t.contains(key) && t.starts_with("P:") {
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

fn model_matrix(t: [f32; 3], rot_deg: [f32; 3], s: [f32; 3]) -> Mat4 {
    let rx = rot_deg[0].to_radians();
    let ry = rot_deg[1].to_radians();
    let rz = rot_deg[2].to_radians();
    let q = Quat::from_euler(glam::EulerRot::XYZ, rx, ry, rz);
    Mat4::from_scale_rotation_translation(Vec3::from(s), q, Vec3::from(t))
}

fn convert_doc(doc: &FbxDoc) -> Result<Scene> {
    let mut scene = Scene::default();
    let u = doc.unit_to_meters;
    // Guard against absurd scales from bad UnitScaleFactor
    let u = if u.is_finite() && u > 1e-9 && u < 1e6 { u } else { 0.01 };

    // Materials
    let mut mat_id_to_index: HashMap<i64, usize> = HashMap::new();
    for (&id, m) in &doc.materials {
        let idx = scene.materials.len();
        mat_id_to_index.insert(id, idx);
        scene.materials.push(Material {
            name: m.name.clone(),
            base_color: [m.diffuse[0], m.diffuse[1], m.diffuse[2], 1.0],
            roughness: 0.5,
            ..Default::default()
        });
    }
    if scene.materials.is_empty() {
        scene.materials.push(Material::default());
    }

    // Map geometry id -> model id (child geom linked to parent model)
    let mut geom_to_model: HashMap<i64, i64> = HashMap::new();
    let mut model_to_mat: HashMap<i64, i64> = HashMap::new();
    for &(child, parent) in &doc.connections {
        if doc.geometries.contains_key(&child) && doc.models.contains_key(&parent) {
            geom_to_model.insert(child, parent);
        }
        if doc.materials.contains_key(&child) && doc.models.contains_key(&parent) {
            model_to_mat.insert(parent, child);
        }
        // sometimes material connected the other way
        if doc.materials.contains_key(&parent) && doc.models.contains_key(&child) {
            model_to_mat.insert(child, parent);
        }
    }

    // Build objects: each geometry that is connected to a model, or every geometry
    let mut used_geom = BTreeSet::new();
    for (&gid, geom) in &doc.geometries {
        let model = geom_to_model
            .get(&gid)
            .and_then(|mid| doc.models.get(mid))
            .cloned()
            .unwrap_or_else(|| Model {
                name: geom.name.clone(),
                kind: "Mesh".into(),
                scale: [1.0; 3],
                ..Default::default()
            });
        let mid = geom_to_model.get(&gid).copied();
        used_geom.insert(gid);

        let mut mesh = geom_to_mesh(geom)?;
        if mesh.indices.is_empty() || mesh.vertices.is_empty() {
            continue;
        }
        // Apply geometric (pivot) transform in file units, then convert to meters
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
        let mut obj = Object::new(model.name.clone(), mesh);

        // Local transform — translation in meters; Lcl Scaling stays as-is (dimensionless)
        obj.translation = Vec3::from(model.translation) * u;
        obj.scale = Vec3::from(model.scale);
        let rx = model.rotation_deg[0].to_radians();
        let ry = model.rotation_deg[1].to_radians();
        let rz = model.rotation_deg[2].to_radians();
        obj.rotation = Quat::from_euler(glam::EulerRot::XYZ, rx, ry, rz);

        if let Some(mid) = mid {
            if let Some(mat_id) = model_to_mat.get(&mid) {
                if let Some(&idx) = mat_id_to_index.get(mat_id) {
                    obj.material = Some(idx);
                }
            }
        }
        scene.objects.push(obj);
    }

    // Orphan geometries not in connections
    for (&gid, geom) in &doc.geometries {
        if used_geom.contains(&gid) {
            continue;
        }
        let mesh = geom_to_mesh(geom)?;
        if mesh.indices.is_empty() {
            continue;
        }
        scene.objects.push(Object::new(geom.name.clone(), mesh));
    }

    if scene.objects.is_empty() {
        bail!("FBX contained geometry but no usable triangles");
    }
    Ok(scene)
}

fn geom_to_mesh(g: &Geom) -> Result<MeshData> {
    // Expand polygons to triangles using FBX polygon vertex index convention
    let mut tri_indices: Vec<u32> = Vec::new();
    let mut poly: Vec<u32> = Vec::new();
    for &idx in &g.polygon_vertex_index {
        if idx < 0 {
            // last vertex of polygon: ~idx
            let v = (!idx) as u32;
            poly.push(v);
            // fan triangulation
            if poly.len() >= 3 {
                for i in 1..poly.len() - 1 {
                    tri_indices.push(poly[0]);
                    tri_indices.push(poly[i]);
                    tri_indices.push(poly[i + 1]);
                }
            }
            poly.clear();
        } else {
            poly.push(idx as u32);
        }
    }

    // If no polygon indices, try sequential triangles from vertex count
    if tri_indices.is_empty() && g.vertices.len() >= 3 {
        for i in (0..g.vertices.len() as u32).step_by(3) {
            if i + 2 < g.vertices.len() as u32 {
                tri_indices.extend_from_slice(&[i, i + 1, i + 2]);
            }
        }
    }

    let n = g.vertices.len();
    let mut vertices = Vec::with_capacity(n);
    for i in 0..n {
        let p = g.vertices[i];
        let normal = g.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
        let uv = if !g.uv_indices.is_empty() && i < g.uv_indices.len() {
            let ui = g.uv_indices[i];
            if ui >= 0 && (ui as usize) < g.uvs.len() {
                let u = g.uvs[ui as usize];
                [u[0], 1.0 - u[1]]
            } else {
                [0.0, 0.0]
            }
        } else if i < g.uvs.len() {
            let u = g.uvs[i];
            [u[0], 1.0 - u[1]]
        } else {
            [0.0, 0.0]
        };
        vertices.push(Vertex {
            position: p,
            normal,
            uv,
        });
    }

    let mut md = MeshData {
        name: g.name.clone(),
        vertices,
        indices: tri_indices,
        groups: Vec::new(),
    };
    if g.normals.is_empty() {
        md.compute_normals();
    }
    Ok(md)
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
        out.push_str("            P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,0,0\n");
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
        let mid = mat_ids.get(mat_i).or_else(|| mat_ids.first()).copied().unwrap_or(0);
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
        Self { data, pos: 0, version }
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
        let v = f64::from_bits(u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap()));
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
        if self.version >= 7500 { 8 } else { 4 }
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
        if let Some(node) = read_node(&mut r)? {
            roots.push(node);
        } else {
            break;
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
        bail!("Binary FBX: no mesh Geometry found (version {version})");
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
        match read_node(r)? {
            Some(n) => children.push(n),
            None => break,
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
    Ok(Some(BinNode { name, props, children }))
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
        dec.read_to_end(&mut out).context("zlib decompress FBX array")?;
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
            if class != "Mesh" && !class.is_empty() && class != "Mesh" {
                // still try if it has Vertices
            }
            let mut g = Geom {
                name: strip_name(&name),
                ..Default::default()
            };
            fill_geom_from_children(node, &mut g);
            if !g.vertices.is_empty() || !g.polygon_vertex_index.is_empty() {
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
            let mut mat = Mat {
                name: strip_name(&name),
                diffuse: [0.8, 0.8, 0.8],
            };
            fill_material_from_children(node, &mut mat);
            doc.materials.insert(id, mat);
        }
        _ => {
            for ch in &node.children {
                ingest_object(ch, doc);
            }
        }
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
                    if n.name == "Normals" {
                        if let Some(BinProp::F64Arr(a)) = n.props.first() {
                            for c in a.chunks_exact(3) {
                                g.normals.push([c[0] as f32, c[1] as f32, c[2] as f32]);
                            }
                        } else if let Some(BinProp::F32Arr(a)) = n.props.first() {
                            for c in a.chunks_exact(3) {
                                g.normals.push([c[0], c[1], c[2]]);
                            }
                        }
                    }
                }
            }
            "LayerElementUV" => {
                for n in &ch.children {
                    if n.name == "UV" {
                        if let Some(BinProp::F64Arr(a)) = n.props.first() {
                            for c in a.chunks_exact(2) {
                                g.uvs.push([c[0] as f32, c[1] as f32]);
                            }
                        } else if let Some(BinProp::F32Arr(a)) = n.props.first() {
                            for c in a.chunks_exact(2) {
                                g.uvs.push([c[0], c[1]]);
                            }
                        }
                    }
                    if n.name == "UVIndex" {
                        if let Some(BinProp::I32Arr(a)) = n.props.first() {
                            g.uv_indices = a.clone();
                        }
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
                if nums.len() >= 3 {
                    let v = [nums[nums.len() - 3], nums[nums.len() - 2], nums[nums.len() - 1]];
                    if key.contains("Lcl Translation") {
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
                if key.contains("UnitScaleFactor") {
                    let nums = prop_trailing_f32(p);
                    if let Some(&v) = nums.last() {
                        doc.unit_to_meters = unit_factor_to_meters(v);
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
                if key.contains("Diffuse") {
                    let nums = prop_trailing_f32(p);
                    if nums.len() >= 3 {
                        mat.diffuse = [nums[nums.len() - 3], nums[nums.len() - 2], nums[nums.len() - 1]];
                    }
                }
            }
        }
    }
}

fn parse_bin_connection(node: &BinNode, doc: &mut FbxDoc) {
    // C: "OO", child, parent  OR typed links
    let mut ids: Vec<i64> = Vec::new();
    for p in &node.props {
        match p {
            BinProp::I64(v) => ids.push(*v),
            BinProp::I32(v) => ids.push(*v as i64),
            _ => {}
        }
    }
    if ids.len() >= 2 {
        doc.connections.push((ids[0], ids[1]));
    }
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

fn build_binary_fbx(scene: &Scene, only: Option<&BTreeSet<u64>>) -> Result<Vec<u8>> {
    let mut meshes: Vec<(String, MeshData, [f32; 4], usize)> = Vec::new(); // name, mesh, diffuse, mat_idx
    let mut mat_names: Vec<(String, [f32; 4])> = scene
        .materials
        .iter()
        .map(|m| (m.name.clone(), m.base_color))
        .collect();
    if mat_names.is_empty() {
        mat_names.push(("Default".into(), [0.8, 0.8, 0.8, 1.0]));
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
        let mi = obj.material.unwrap_or(0).min(mat_names.len().saturating_sub(1));
        let diff = mat_names.get(mi).map(|(_, c)| *c).unwrap_or([0.8, 0.8, 0.8, 1.0]);
        meshes.push((obj.name.clone(), baked, diff, mi));
    }
    if meshes.is_empty() {
        bail!("nothing to export as FBX (no mesh objects)");
    }

    // Collect unique materials referenced
    let mut used_mats: Vec<(String, [f32; 4])> = Vec::new();
    let mut mat_remap: Vec<usize> = vec![0; mat_names.len()];
    for (_, _, _, mi) in &meshes {
        if used_mats.iter().all(|(n, _)| n != &mat_names[*mi].0) {
            mat_remap[*mi] = used_mats.len();
            used_mats.push(mat_names[*mi].clone());
        } else {
            mat_remap[*mi] = used_mats.iter().position(|(n, _)| n == &mat_names[*mi].0).unwrap_or(0);
        }
    }
    if used_mats.is_empty() {
        used_mats.push(("Default".into(), [0.8, 0.8, 0.8, 1.0]));
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

    let definitions = bin_node(
        "Definitions",
        vec![],
        vec![
            bin_node("Version", vec![BinWProp::I32(100)], vec![]),
            bin_node("Count", vec![BinWProp::I32(3)], vec![]),
            object_type_def("Geometry", meshes.len() as i32),
            object_type_def("Model", meshes.len() as i32),
            object_type_def("Material", used_mats.len() as i32),
        ],
    );

    let mut object_children: Vec<BinWNode> = Vec::new();
    for (i, (name, color)) in used_mats.iter().enumerate() {
        object_children.push(material_node(mat_ids[i], name, *color));
    }
    for (i, (name, mesh, _, mi)) in meshes.iter().enumerate() {
        object_children.push(geometry_node(geom_ids[i], name, mesh));
        object_children.push(model_node(model_ids[i], name));
        let _ = mi;
    }
    let objects = bin_node("Objects", vec![], object_children);

    let mut conn_children: Vec<BinWNode> = Vec::new();
    for i in 0..meshes.len() {
        // Geometry -> Model
        conn_children.push(connection_oo(geom_ids[i], model_ids[i]));
        // Model -> root (0)
        conn_children.push(connection_oo(model_ids[i], 0));
        let mi = meshes[i].3;
        let mid = mat_ids[mat_remap.get(mi).copied().unwrap_or(0).min(mat_ids.len() - 1)];
        conn_children.push(connection_oo(mid, model_ids[i]));
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
            bin_node("ShadingModel", vec![BinWProp::String("Phong".into())], vec![]),
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
    for v in &mesh.vertices {
        verts.extend_from_slice(&[v.position[0] as f64, v.position[1] as f64, v.position[2] as f64]);
        norms.extend_from_slice(&[v.normal[0] as f64, v.normal[1] as f64, v.normal[2] as f64]);
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
            bin_node("Culling", vec![BinWProp::String("CullingOff".into())], vec![]),
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

        let bytes = build_binary_fbx(&scene, None).expect("export binary");
        assert!(bytes.starts_with(b"Kaydara FBX Binary"), "missing binary magic");
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
        assert!(
            (max_abs - 0.5).abs() < 0.05,
            "expected 0.5m, got {max_abs}"
        );
    }
}
