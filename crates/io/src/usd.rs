//! Minimal USD ASCII (.usda) exporter.
//!
//! USD is text-based and much simpler to write correctly than Alembic/Ogawa.
//! Blender (2.83+) imports .usda via File → Import → Universal Scene Description.
//!
//! Supports: mesh objects, world transforms, and mesh-sequence (fluid) animation
//! via time-sampled `points` / `faceVertexIndices` / `faceVertexCounts`.

use anyhow::{bail, Result};
use forge_core::{scene_at_frame, MeshData, Scene};
use glam::Mat4;
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

fn safe_ident(raw: &str) -> String {
    let mut s: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() || s.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        s = format!("Mesh_{s}");
    }
    s
}

fn mat4_to_usd(m: Mat4) -> String {
    // USD xformOp:transform is row-major 4x4.
    let a = m.transpose().to_cols_array();
    format!(
        "({}, {}, {}, {}), ({}, {}, {}, {}), ({}, {}, {}, {}), ({}, {}, {}, {})",
        a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
        a[14], a[15]
    )
}

fn mesh_arrays(mesh: &MeshData) -> (String, String, String) {
    let mut pts = String::from("[");
    for (i, v) in mesh.vertices.iter().enumerate() {
        if i > 0 {
            pts.push(',');
        }
        let p = v.position;
        let x = if p[0].is_finite() { p[0] } else { 0.0 };
        let y = if p[1].is_finite() { p[1] } else { 0.0 };
        let z = if p[2].is_finite() { p[2] } else { 0.0 };
        let _ = write!(pts, "({x}, {y}, {z})");
    }
    pts.push(']');

    let nvert = mesh.vertices.len();
    let mut counts = String::from("[");
    let mut indices = String::from("[");
    let mut first_face = true;
    let mut first_idx = true;
    for t in mesh.indices.chunks_exact(3) {
        if (t[0] as usize) >= nvert || (t[1] as usize) >= nvert || (t[2] as usize) >= nvert {
            continue;
        }
        if !first_face {
            counts.push(',');
        }
        first_face = false;
        counts.push('3');
        for &ix in t {
            if !first_idx {
                indices.push(',');
            }
            first_idx = false;
            let _ = write!(indices, "{ix}");
        }
    }
    counts.push(']');
    indices.push(']');
    (pts, counts, indices)
}

/// Build a .usda document for the scene (or a selection of object ids).
pub fn build_usda_string(scene: &Scene, only: Option<&BTreeSet<u64>>) -> Result<String> {
    let fps = scene.animation.fps.max(1) as f64;
    let start = scene.animation.frame_start;
    let end = scene.animation.frame_end.max(start);

    let exported: Vec<usize> = (0..scene.objects.len())
        .filter(|&i| {
            let o = &scene.objects[i];
            scene.is_visible(i)
                && o.kind.is_mesh()
                && only.map_or(true, |s| s.contains(&o.id))
                && (!o.mesh.indices.is_empty() || scene.mesh_caches.contains_key(&o.id))
        })
        .collect();
    if exported.is_empty() {
        bail!("nothing to export: no visible mesh objects");
    }

    let any_cache = exported
        .iter()
        .any(|&i| scene.mesh_caches.contains_key(&scene.objects[i].id));
    let any_motion = !scene.animation.objects.is_empty() || scene.physics_cache.is_some();
    let animated = any_cache || any_motion;
    let frames: Vec<i32> = if animated {
        (start..=end).collect()
    } else {
        vec![scene.animation.frame_current]
    };

    let mut out = String::new();
    out.push_str("#usda 1.0\n");
    out.push_str("(\n");
    let _ = writeln!(out, "    defaultPrim = \"World\"");
    let _ = writeln!(out, "    metersPerUnit = 1");
    let _ = writeln!(out, "    upAxis = \"Y\"");
    if animated {
        let _ = writeln!(out, "    startTimeCode = {start}");
        let _ = writeln!(out, "    endTimeCode = {end}");
        let _ = writeln!(out, "    timeCodesPerSecond = {fps}");
        let _ = writeln!(out, "    framesPerSecond = {fps}");
    }
    out.push_str(")\n\n");
    out.push_str("def Xform \"World\"\n{\n");

    let mut used = std::collections::HashSet::new();
    for &i in &exported {
        let o = &scene.objects[i];
        let mut name = safe_ident(&o.name);
        let mut n = 1;
        while !used.insert(name.clone()) {
            name = format!("{}_{n}", safe_ident(&o.name));
            n += 1;
        }

        let cache = scene.mesh_caches.get(&o.id);
        let geo_animated = cache.map_or(false, |c| c.len() > 1) && frames.len() > 1;

        let _ = writeln!(out, "    def Xform \"{name}\"");
        out.push_str("    {\n");
        // Static transform from current/first evaluated frame.
        let eval0 = scene_at_frame(scene, frames[0]);
        let worlds = eval0.world_matrices();
        let m = worlds.get(i).copied().unwrap_or(Mat4::IDENTITY);
        let _ = writeln!(
            out,
            "        matrix4d xformOp:transform = ({})",
            mat4_to_usd(m)
        );
        out.push_str("        uniform token[] xformOpOrder = [\"xformOp:transform\"]\n\n");

        let _ = writeln!(out, "        def Mesh \"{name}Shape\"");
        out.push_str("        {\n");

        if geo_animated {
            out.push_str("            point3f[] points.timeSamples = {\n");
            for &f in &frames {
                let mesh: Arc<MeshData> = cache
                    .and_then(|c| c.mesh_at(f as f32, scene.animation.fps as f32))
                    .unwrap_or_else(|| o.mesh.clone());
                let (pts, _, _) = mesh_arrays(&mesh);
                let _ = writeln!(out, "                {f}: {pts},");
            }
            out.push_str("            }\n");
            out.push_str("            int[] faceVertexCounts.timeSamples = {\n");
            for &f in &frames {
                let mesh: Arc<MeshData> = cache
                    .and_then(|c| c.mesh_at(f as f32, scene.animation.fps as f32))
                    .unwrap_or_else(|| o.mesh.clone());
                let (_, counts, _) = mesh_arrays(&mesh);
                let _ = writeln!(out, "                {f}: {counts},");
            }
            out.push_str("            }\n");
            out.push_str("            int[] faceVertexIndices.timeSamples = {\n");
            for &f in &frames {
                let mesh: Arc<MeshData> = cache
                    .and_then(|c| c.mesh_at(f as f32, scene.animation.fps as f32))
                    .unwrap_or_else(|| o.mesh.clone());
                let (_, _, indices) = mesh_arrays(&mesh);
                let _ = writeln!(out, "                {f}: {indices},");
            }
            out.push_str("            }\n");
        } else {
            let mesh = cache
                .and_then(|c| c.mesh_at(frames[0] as f32, scene.animation.fps as f32))
                .unwrap_or_else(|| o.mesh.clone());
            let (pts, counts, indices) = mesh_arrays(&mesh);
            let _ = writeln!(out, "            point3f[] points = {pts}");
            let _ = writeln!(out, "            int[] faceVertexCounts = {counts}");
            let _ = writeln!(out, "            int[] faceVertexIndices = {indices}");
        }
        out.push_str("            uniform token subdivisionScheme = \"none\"\n");
        out.push_str("        }\n");
        out.push_str("    }\n");
    }

    out.push_str("}\n");
    Ok(out)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn export_usda(scene: &Scene, path: &Path, only: Option<&BTreeSet<u64>>) -> Result<()> {
    let s = build_usda_string(scene, only)?;
    std::fs::write(path, s).map_err(|e| anyhow::anyhow!("writing {}: {e}", path.display()))
}

#[cfg(target_arch = "wasm32")]
pub fn export_usda(_scene: &Scene, _path: &Path, _only: Option<&BTreeSet<u64>>) -> Result<()> {
    bail!("USD export is not available on WebAssembly")
}
