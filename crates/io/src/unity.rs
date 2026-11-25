//! Unity `.unitypackage` import.
//!
//! A `.unitypackage` is a gzip-compressed tar archive. Every asset lives in its own
//! folder named after the asset GUID and holds `pathname` (the project path),
//! `asset` (the raw file) and `asset.meta`. We pull out every embedded model
//! (`.fbx`, `.obj`, `.glb`) and run it through the regular importers.
//!
//! Textures and `.mtl` files that sit in the package next to a model are found by file name
//! (FBX / OBJ). Not supported: Unity-native serialized assets (`.asset`, `.prefab`, `.unity`,
//! `.mat`) and animation clips.
use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use forge_core::Scene;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

/// Refuse to inflate absurdly large packages into RAM.
const MAX_UNPACKED_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Default)]
struct Entry {
    pathname: Option<String>,
    /// Byte range of the `asset` file inside the decompressed archive.
    asset: Option<(usize, usize)>,
}

fn c_string(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

fn octal(field: &[u8]) -> usize {
    let digits: String = field
        .iter()
        .map(|&b| b as char)
        .filter(|c| ('0'..='7').contains(c))
        .collect();
    usize::from_str_radix(&digits, 8).unwrap_or(0)
}

/// Walk the tar archive and group entries by asset GUID.
fn index_tar(data: &[u8]) -> Result<BTreeMap<String, Entry>> {
    let mut entries: BTreeMap<String, Entry> = BTreeMap::new();
    let mut pos = 0usize;
    let mut long_name: Option<String> = None;
    while pos + 512 <= data.len() {
        let h = &data[pos..pos + 512];
        if h.iter().all(|&b| b == 0) {
            break;
        }
        let mut name = c_string(&h[0..100]);
        let size = octal(&h[124..136]);
        let kind = h[156];
        if &h[257..262] == b"ustar" {
            let prefix = c_string(&h[345..500]);
            if !prefix.is_empty() {
                name = format!("{prefix}/{name}");
            }
        }
        let start = pos + 512;
        let end = start
            .checked_add(size)
            .filter(|&e| e <= data.len())
            .context("unitypackage is truncated or corrupt")?;
        pos = start + size.div_ceil(512) * 512;

        match kind {
            // GNU long file name: the body is the name of the next entry.
            b'L' => {
                long_name = Some(c_string(&data[start..end]));
                continue;
            }
            b'0' | 0 => {
                let name = long_name.take().unwrap_or(name);
                let name = name.trim_start_matches("./");
                let Some((guid, rest)) = name.split_once('/') else {
                    continue;
                };
                let entry = entries.entry(guid.to_string()).or_default();
                match rest {
                    "pathname" => {
                        let text = String::from_utf8_lossy(&data[start..end]);
                        let line = text.lines().next().unwrap_or("").trim();
                        if !line.is_empty() {
                            entry.pathname = Some(line.to_string());
                        }
                    }
                    "asset" => entry.asset = Some((start, end)),
                    _ => {}
                }
            }
            _ => long_name = None,
        }
    }
    Ok(entries)
}

fn ext_lower(path: &str) -> String {
    path.rsplit('.').next().unwrap_or("").to_ascii_lowercase()
}

fn file_stem(path: &str) -> String {
    let file = path.rsplit('/').next().unwrap_or(path);
    file.rsplit_once('.').map(|(s, _)| s).unwrap_or(file).to_string()
}

/// Files other assets can refer to by name (textures, .mtl): lower-case file name -> bytes range.
type SideFiles = BTreeMap<String, (usize, usize)>;

fn is_side_file(ext: &str) -> bool {
    matches!(ext, "png" | "jpg" | "jpeg" | "tga" | "bmp" | "mtl")
}

fn lower_file_name(path: &str) -> String {
    let norm = path.replace('\\', "/");
    norm.rsplit('/').next().unwrap_or(&norm).to_ascii_lowercase()
}

fn bare_name(path: &str) -> String {
    let norm = path.replace('\\', "/");
    norm.rsplit('/').next().unwrap_or(&norm).to_string()
}

fn side_file<'a>(data: &'a [u8], side: &SideFiles, name: &str) -> Option<&'a [u8]> {
    let (a, b) = *side.get(&lower_file_name(name))?;
    Some(&data[a..b])
}

/// Names of the files an OBJ's `mtllib` / an MTL's `map_*` lines refer to.
fn referenced_names(text: &str, keys: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some((key, rest)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        if keys.iter().any(|k| key.eq_ignore_ascii_case(k)) {
            // texture options (-s 1 1 1 ...) precede the name; the name is the last token
            if let Some(last) = rest.split_whitespace().last() {
                out.push(last.to_string());
            }
        }
    }
    out
}

/// Load a model whose importer only accepts a path, via a temporary folder. Any `.mtl` and
/// textures the model refers to are copied next to it so the importer finds them.
fn load_via_temp(
    bytes: &[u8],
    file_name: &str,
    data: &[u8],
    side: &SideFiles,
    load: impl FnOnce(&Path) -> Result<Scene>,
) -> Result<Scene> {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("blendcraft_unity_{}_{n}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(file_name);
    std::fs::write(&path, bytes)?;
    if file_name.ends_with(".obj") {
        let text = String::from_utf8_lossy(bytes);
        for mtl in referenced_names(&text, &["mtllib"]) {
            let Some(mb) = side_file(data, side, &mtl) else {
                continue;
            };
            let _ = std::fs::write(dir.join(bare_name(&mtl)), mb);
            let mtl_text = String::from_utf8_lossy(mb);
            let keys = ["map_Kd", "map_Bump", "bump", "map_Ke", "map_d", "norm"];
            for tex in referenced_names(&mtl_text, &keys) {
                if let Some(tb) = side_file(data, side, &tex) {
                    let _ = std::fs::write(dir.join(bare_name(&tex)), tb);
                }
            }
        }
    }
    let result = load(&path);
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// Import all models contained in a `.unitypackage`.
///
/// Returns the loaded models as `(unity path, scene)` plus a list of human-readable
/// notes for assets that were skipped or failed to load.
pub fn load_unitypackage(path: &Path) -> Result<(Vec<(String, Scene)>, Vec<String>)> {
    let file =
        std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut data = Vec::new();
    GzDecoder::new(file)
        .take(MAX_UNPACKED_BYTES + 1)
        .read_to_end(&mut data)
        .context("not a valid .unitypackage (gzip data could not be read)")?;
    if data.len() as u64 > MAX_UNPACKED_BYTES {
        bail!("unitypackage is larger than 2 GiB once unpacked");
    }

    let entries = index_tar(&data)?;
    // Textures / .mtl files are separate assets in the package; index them by file name so
    // models that refer to them by name can find them.
    let mut side: SideFiles = BTreeMap::new();
    for entry in entries.values() {
        if let (Some(pathname), Some(range)) = (&entry.pathname, entry.asset) {
            if is_side_file(&ext_lower(pathname)) {
                side.entry(lower_file_name(pathname)).or_insert(range);
            }
        }
    }
    let mut models = Vec::new();
    let mut notes = Vec::new();
    for entry in entries.values() {
        let (Some(pathname), Some((a, b))) = (&entry.pathname, entry.asset) else {
            continue;
        };
        let bytes = &data[a..b];
        let loaded = match ext_lower(pathname).as_str() {
            "fbx" => crate::load_fbx_bytes_with(bytes, &|name: &str| {
                side_file(&data, &side, name).map(|b| b.to_vec())
            }),
            "obj" => load_via_temp(bytes, "model.obj", &data, &side, crate::load_obj),
            "glb" => load_via_temp(bytes, "model.glb", &data, &side, crate::load_gltf),
            _ => continue,
        };
        match loaded {
            Ok(mut scene) if !scene.objects.is_empty() => {
                // Name single-mesh models after the Unity file for easy recognition.
                if scene.objects.len() == 1 {
                    scene.objects[0].name = file_stem(pathname);
                }
                models.push((pathname.clone(), scene));
            }
            Ok(_) => notes.push(format!("{pathname}: contains no meshes")),
            Err(e) => notes.push(format!("{pathname}: {e:#}")),
        }
    }
    models.sort_by(|a, b| a.0.cmp(&b.0));
    Ok((models, notes))
}
