//! Unity `.unitypackage` import.
//!
//! A `.unitypackage` is a gzip-compressed tar archive. Every asset lives in its own
//! folder named after the asset GUID and holds `pathname` (the project path),
//! `asset` (the raw file) and `asset.meta`. We pull out every embedded model
//! (`.fbx`, `.obj`, `.glb`) and run it through the regular importers.
//!
//! Not supported: Unity-native serialized assets (`.asset`, `.prefab`, `.unity`,
//! `.mat`), textures that live beside the model, and animation clips.
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

/// Load a model whose importer only accepts a path, via a temporary file.
fn load_via_temp(
    bytes: &[u8],
    file_name: &str,
    load: impl FnOnce(&Path) -> Result<Scene>,
) -> Result<Scene> {
    let dir = std::env::temp_dir().join(format!("blendcraft_unity_{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(file_name);
    std::fs::write(&path, bytes)?;
    let result = load(&path);
    let _ = std::fs::remove_file(&path);
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
    let mut models = Vec::new();
    let mut notes = Vec::new();
    for entry in entries.values() {
        let (Some(pathname), Some((a, b))) = (&entry.pathname, entry.asset) else {
            continue;
        };
        let bytes = &data[a..b];
        let loaded = match ext_lower(pathname).as_str() {
            "fbx" => crate::load_fbx_bytes(bytes),
            "obj" => load_via_temp(bytes, "model.obj", crate::load_obj),
            "glb" => load_via_temp(bytes, "model.glb", crate::load_gltf),
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
