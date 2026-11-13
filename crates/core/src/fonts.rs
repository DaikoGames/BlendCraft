//! TrueType/OpenType font loading and text → mesh (local files only).
//! Fonts are read from bytes (disk or one-time downloaded cache); no font service APIs.
use crate::mesh::{MeshData, Vertex};
use lyon::math::Point;
use lyon::path::Path;
use lyon::tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use ttf_parser::{Face, OutlineBuilder};

/// Global in-process font registry (name → file bytes).
static FONT_REG: OnceLock<Mutex<FontRegistry>> = OnceLock::new();

pub fn font_registry() -> &'static Mutex<FontRegistry> {
    FONT_REG.get_or_init(|| Mutex::new(FontRegistry::default()))
}

#[derive(Default)]
pub struct FontRegistry {
    /// Display name → font file bytes (TTF/OTF)
    fonts: HashMap<String, Arc<Vec<u8>>>,
}

impl FontRegistry {
    pub fn insert(&mut self, name: impl Into<String>, bytes: Vec<u8>) -> Result<(), String> {
        let name = name.into();
        Face::parse(&bytes, 0).map_err(|e| format!("invalid font '{name}': {e:?}"))?;
        self.fonts.insert(name, Arc::new(bytes));
        Ok(())
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<_> = self.fonts.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn get(&self, name: &str) -> Option<Arc<Vec<u8>>> {
        self.fonts.get(name).cloned()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.fonts.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.fonts.len()
    }
}

/// Build a mesh for `text` using a registered font (by name) or return None if missing.
pub fn mesh_from_font(
    font_name: &str,
    text: &str,
    size: f32,
    extrude: bool,
    depth: f32,
) -> Option<MeshData> {
    let reg = font_registry().lock().ok()?;
    let bytes = reg.get(font_name)?;
    drop(reg);
    mesh_from_font_bytes(&bytes, text, size, extrude, depth).ok()
}

pub fn mesh_from_font_bytes(
    font_bytes: &[u8],
    text: &str,
    size: f32,
    extrude: bool,
    depth: f32,
) -> Result<MeshData, String> {
    let face = Face::parse(font_bytes, 0).map_err(|e| format!("parse font: {e:?}"))?;
    let units = face.units_per_em() as f32;
    if units <= 0.0 {
        return Err("invalid units_per_em".into());
    }
    let scale = size / units;
    let depth = if extrude {
        depth.max(0.01)
    } else {
        size * 0.02
    };
    let line_gap = face.line_gap() as f32 * scale;
    let asc = face.ascender() as f32 * scale;
    let line_height = (face.ascender() as f32 - face.descender() as f32) * scale + line_gap;

    let mut all_verts: Vec<Vertex> = Vec::new();
    let mut all_idx: Vec<u32> = Vec::new();
    let mut pen_x = 0.0f32;
    let mut pen_y = 0.0f32;
    let mut max_x = 0.0f32;

    let text = if text.is_empty() { " " } else { text };

    for ch in text.chars() {
        if ch == '\n' {
            pen_x = 0.0;
            pen_y -= line_height;
            continue;
        }
        let Some(gid) = face.glyph_index(ch) else {
            pen_x += size * 0.4;
            continue;
        };
        let adv = face.glyph_hor_advance(gid).unwrap_or(0) as f32 * scale;

        let mut builder = LyonOutline::default();
        face.outline_glyph(gid, &mut builder);
        if let Some(path) = builder.into_path() {
            if let Ok(buffers) = tessellate_path(&path) {
                let z0 = -depth * 0.5;
                let z1 = depth * 0.5;
                let base = all_verts.len() as u32;

                // Front face (z = z1)
                for v in &buffers.vertices {
                    let x = pen_x + v.x * scale;
                    let y = pen_y + v.y * scale;
                    max_x = max_x.max(x);
                    all_verts.push(vert([x, y, z1], [0.0, 0.0, 1.0]));
                }
                for tri in buffers.indices.chunks_exact(3) {
                    all_idx.extend_from_slice(&[
                        base + tri[0] as u32,
                        base + tri[1] as u32,
                        base + tri[2] as u32,
                    ]);
                }

                if extrude {
                    // Back face (z = z0), reversed winding
                    let back = all_verts.len() as u32;
                    for v in &buffers.vertices {
                        let x = pen_x + v.x * scale;
                        let y = pen_y + v.y * scale;
                        all_verts.push(vert([x, y, z0], [0.0, 0.0, -1.0]));
                    }
                    for tri in buffers.indices.chunks_exact(3) {
                        all_idx.extend_from_slice(&[
                            back + tri[0] as u32,
                            back + tri[2] as u32,
                            back + tri[1] as u32,
                        ]);
                    }

                    // Side walls from outline edges (approximate using tessellation boundary is hard;
                    // extrude each triangle edge that is on boundary is complex).
                    // Simpler: for each triangle edge, skip internal; use path flattening for sides.
                    add_side_walls(
                        &mut all_verts,
                        &mut all_idx,
                        &path,
                        pen_x,
                        pen_y,
                        scale,
                        z0,
                        z1,
                    );
                }
            }
        }
        pen_x += adv;
        max_x = max_x.max(pen_x);
    }

    // Center horizontally; shift so baseline sits near origin
    let shift_x = -max_x * 0.5;
    let shift_y = -asc * 0.15;
    for v in &mut all_verts {
        v.position[0] += shift_x;
        v.position[1] += shift_y;
    }

    if all_verts.is_empty() {
        // empty glyph fallback: small quad
        return Ok(MeshData {
            name: "Text".into(),
            vertices: vec![
                vert([-size * 0.1, 0.0, 0.0], [0.0, 0.0, 1.0]),
                vert([size * 0.1, 0.0, 0.0], [0.0, 0.0, 1.0]),
                vert([size * 0.1, size * 0.2, 0.0], [0.0, 0.0, 1.0]),
                vert([-size * 0.1, size * 0.2, 0.0], [0.0, 0.0, 1.0]),
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            ..Default::default()
        });
    }

    Ok(MeshData {
        name: "Text".into(),
        vertices: all_verts,
        indices: all_idx,
        ..Default::default()
    })
}

fn vert(p: [f32; 3], n: [f32; 3]) -> Vertex {
    Vertex {
        position: p,
        normal: n,
        uv: [0.0, 0.0],
    }
}

fn tessellate_path(path: &Path) -> Result<VertexBuffers<Point, u16>, String> {
    let mut buffers: VertexBuffers<Point, u16> = VertexBuffers::new();
    let mut tess = FillTessellator::new();
    tess.tessellate_path(
        path,
        &FillOptions::default(),
        &mut BuffersBuilder::new(&mut buffers, |vertex: FillVertex| vertex.position()),
    )
    .map_err(|e| format!("tessellate: {e:?}"))?;
    Ok(buffers)
}

/// Extrude the flattened path as side walls.
fn add_side_walls(
    verts: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    path: &Path,
    pen_x: f32,
    pen_y: f32,
    scale: f32,
    z0: f32,
    z1: f32,
) {
    // Flatten path to line segments
    let mut pts: Vec<(f32, f32)> = Vec::new();
    let mut last = (0.0f32, 0.0f32);
    for e in path.iter() {
        match e {
            lyon::path::PathEvent::Begin { at } => {
                last = (at.x, at.y);
                pts.push(last);
            }
            lyon::path::PathEvent::Line { to, .. } => {
                last = (to.x, to.y);
                pts.push(last);
            }
            lyon::path::PathEvent::Quadratic { ctrl, to, .. } => {
                // subdivide
                for i in 1..=4 {
                    let t = i as f32 / 4.0;
                    let x = (1.0 - t) * (1.0 - t) * last.0
                        + 2.0 * (1.0 - t) * t * ctrl.x
                        + t * t * to.x;
                    let y = (1.0 - t) * (1.0 - t) * last.1
                        + 2.0 * (1.0 - t) * t * ctrl.y
                        + t * t * to.y;
                    last = (x, y);
                    pts.push(last);
                }
            }
            lyon::path::PathEvent::Cubic {
                ctrl1, ctrl2, to, ..
            } => {
                for i in 1..=4 {
                    let t = i as f32 / 4.0;
                    let u = 1.0 - t;
                    let x = u * u * u * last.0
                        + 3.0 * u * u * t * ctrl1.x
                        + 3.0 * u * t * t * ctrl2.x
                        + t * t * t * to.x;
                    let y = u * u * u * last.1
                        + 3.0 * u * u * t * ctrl1.y
                        + 3.0 * u * t * t * ctrl2.y
                        + t * t * t * to.y;
                    last = (x, y);
                    pts.push(last);
                }
            }
            lyon::path::PathEvent::End { .. } => {
                // close wall strip between consecutive points
                if pts.len() >= 2 {
                    for i in 0..pts.len() - 1 {
                        let (x0, y0) = pts[i];
                        let (x1, y1) = pts[i + 1];
                        let ax = pen_x + x0 * scale;
                        let ay = pen_y + y0 * scale;
                        let bx = pen_x + x1 * scale;
                        let by = pen_y + y1 * scale;
                        let dx = bx - ax;
                        let dy = by - ay;
                        let len = (dx * dx + dy * dy).sqrt().max(1e-8);
                        let nx = dy / len;
                        let ny = -dx / len;
                        let i0 = verts.len() as u32;
                        verts.push(vert([ax, ay, z0], [nx, ny, 0.0]));
                        verts.push(vert([bx, by, z0], [nx, ny, 0.0]));
                        verts.push(vert([bx, by, z1], [nx, ny, 0.0]));
                        verts.push(vert([ax, ay, z1], [nx, ny, 0.0]));
                        indices.extend_from_slice(&[i0, i0 + 1, i0 + 2, i0, i0 + 2, i0 + 3]);
                    }
                }
                pts.clear();
            }
        }
    }
}

#[derive(Default)]
struct LyonOutline {
    builder: lyon::path::Builder,
    started: bool,
}

impl LyonOutline {
    fn into_path(self) -> Option<Path> {
        if self.started {
            Some(self.builder.build())
        } else {
            None
        }
    }
}

impl OutlineBuilder for LyonOutline {
    fn move_to(&mut self, x: f32, y: f32) {
        self.builder.begin(Point::new(x, y));
        self.started = true;
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.builder.line_to(Point::new(x, y));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.builder
            .quadratic_bezier_to(Point::new(x1, y1), Point::new(x, y));
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.builder
            .cubic_bezier_to(Point::new(x1, y1), Point::new(x2, y2), Point::new(x, y));
    }
    fn close(&mut self) {
        self.builder.close();
    }
}

/// Well-known open-licensed fonts (direct .ttf URLs — one-time download, then local only).
pub struct CatalogFont {
    pub name: &'static str,
    pub family: &'static str,
    pub url: &'static str,
    pub license: &'static str,
}

pub fn free_font_catalog() -> &'static [CatalogFont] {
    // Direct file URLs only (GitHub raw / jsDelivr), not a font API.
    &[
        CatalogFont { name: "Roboto Regular", family: "Roboto", url: "https://cdn.jsdelivr.net/fontsource/fonts/roboto@5.0.8/latin-400-normal.ttf", license: "Apache 2.0" },
        CatalogFont { name: "Roboto Bold", family: "Roboto", url: "https://cdn.jsdelivr.net/fontsource/fonts/roboto@5.0.8/latin-700-normal.ttf", license: "Apache 2.0" },
        CatalogFont { name: "Roboto Italic", family: "Roboto", url: "https://cdn.jsdelivr.net/fontsource/fonts/roboto@5.0.8/latin-400-italic.ttf", license: "Apache 2.0" },
        CatalogFont { name: "Open Sans Regular", family: "Open Sans", url: "https://cdn.jsdelivr.net/fontsource/fonts/open-sans@5.0.28/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Open Sans Bold", family: "Open Sans", url: "https://cdn.jsdelivr.net/fontsource/fonts/open-sans@5.0.28/latin-700-normal.ttf", license: "OFL" },
        CatalogFont { name: "Noto Sans Regular", family: "Noto Sans", url: "https://cdn.jsdelivr.net/fontsource/fonts/noto-sans@5.0.22/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Noto Sans Bold", family: "Noto Sans", url: "https://cdn.jsdelivr.net/fontsource/fonts/noto-sans@5.0.22/latin-700-normal.ttf", license: "OFL" },
        CatalogFont { name: "Noto Serif Regular", family: "Noto Serif", url: "https://cdn.jsdelivr.net/fontsource/fonts/noto-serif@5.0.18/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Montserrat Regular", family: "Montserrat", url: "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@5.0.18/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Montserrat Bold", family: "Montserrat", url: "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@5.0.18/latin-700-normal.ttf", license: "OFL" },
        CatalogFont { name: "Lato Regular", family: "Lato", url: "https://cdn.jsdelivr.net/fontsource/fonts/lato@5.0.21/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Lato Bold", family: "Lato", url: "https://cdn.jsdelivr.net/fontsource/fonts/lato@5.0.21/latin-700-normal.ttf", license: "OFL" },
        CatalogFont { name: "Oswald Regular", family: "Oswald", url: "https://cdn.jsdelivr.net/fontsource/fonts/oswald@5.0.20/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Raleway Regular", family: "Raleway", url: "https://cdn.jsdelivr.net/fontsource/fonts/raleway@5.0.18/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Playfair Display Regular", family: "Playfair Display", url: "https://cdn.jsdelivr.net/fontsource/fonts/playfair-display@5.0.24/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Merriweather Regular", family: "Merriweather", url: "https://cdn.jsdelivr.net/fontsource/fonts/merriweather@5.0.21/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Source Code Pro Regular", family: "Source Code Pro", url: "https://cdn.jsdelivr.net/fontsource/fonts/source-code-pro@5.0.18/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Fira Code Regular", family: "Fira Code", url: "https://cdn.jsdelivr.net/fontsource/fonts/fira-code@5.0.18/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Inter Regular", family: "Inter", url: "https://cdn.jsdelivr.net/fontsource/fonts/inter@5.0.18/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Inter Bold", family: "Inter", url: "https://cdn.jsdelivr.net/fontsource/fonts/inter@5.0.18/latin-700-normal.ttf", license: "OFL" },
        CatalogFont { name: "Poppins Regular", family: "Poppins", url: "https://cdn.jsdelivr.net/fontsource/fonts/poppins@5.0.14/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Poppins Bold", family: "Poppins", url: "https://cdn.jsdelivr.net/fontsource/fonts/poppins@5.0.14/latin-700-normal.ttf", license: "OFL" },
        CatalogFont { name: "Ubuntu Regular", family: "Ubuntu", url: "https://cdn.jsdelivr.net/fontsource/fonts/ubuntu@5.0.18/latin-400-normal.ttf", license: "Ubuntu Font Licence" },
        CatalogFont { name: "Rubik Regular", family: "Rubik", url: "https://cdn.jsdelivr.net/fontsource/fonts/rubik@5.0.20/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Work Sans Regular", family: "Work Sans", url: "https://cdn.jsdelivr.net/fontsource/fonts/work-sans@5.0.18/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Nunito Regular", family: "Nunito", url: "https://cdn.jsdelivr.net/fontsource/fonts/nunito@5.0.18/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "PT Sans Regular", family: "PT Sans", url: "https://cdn.jsdelivr.net/fontsource/fonts/pt-sans@5.0.13/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "PT Serif Regular", family: "PT Serif", url: "https://cdn.jsdelivr.net/fontsource/fonts/pt-serif@5.0.13/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "Inconsolata Regular", family: "Inconsolata", url: "https://cdn.jsdelivr.net/fontsource/fonts/inconsolata@5.0.18/latin-400-normal.ttf", license: "OFL" },
        CatalogFont { name: "JetBrains Mono Regular", family: "JetBrains Mono", url: "https://cdn.jsdelivr.net/fontsource/fonts/jetbrains-mono@5.0.21/latin-400-normal.ttf", license: "OFL" },
    ]
}

/// Typical OS font directories (read-only scan).
pub fn system_font_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();
    if cfg!(target_os = "windows") {
        if let Some(w) = std::env::var_os("WINDIR") {
            dirs.push(std::path::PathBuf::from(w).join("Fonts"));
        }
    } else if cfg!(target_os = "macos") {
        dirs.push(std::path::PathBuf::from("/System/Library/Fonts"));
        dirs.push(std::path::PathBuf::from("/Library/Fonts"));
        if let Some(h) = std::env::var_os("HOME") {
            dirs.push(std::path::PathBuf::from(h).join("Library/Fonts"));
        }
    } else {
        dirs.push(std::path::PathBuf::from("/usr/share/fonts"));
        dirs.push(std::path::PathBuf::from("/usr/local/share/fonts"));
        if let Some(h) = std::env::var_os("HOME") {
            dirs.push(std::path::PathBuf::from(&h).join(".fonts"));
            dirs.push(std::path::PathBuf::from(&h).join(".local/share/fonts"));
        }
    }
    dirs
}
