//! Local font files: system fonts and user-selected disk imports. Offline by design.
use forge_core::{font_registry, system_font_dirs};
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
pub fn fonts_dir() -> Option<PathBuf> {
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)?
    } else if cfg!(target_os = "macos") {
        PathBuf::from(std::env::var_os("HOME")?)
            .join("Library")
            .join("Application Support")
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?
    };
    let dir = base.join("BlendCraft").join("fonts");
    let _ = std::fs::create_dir_all(&dir);
    Some(dir)
}

#[cfg(target_arch = "wasm32")]
pub fn fonts_dir() -> Option<PathBuf> {
    None
}

fn try_register(name: &str, path: &Path) {
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    if bytes.len() < 16 {
        return;
    }
    if let Ok(mut reg) = font_registry().lock() {
        if !reg.contains(name) {
            let _ = reg.insert(name.to_string(), bytes);
        }
    }
}

/// Load cached downloads + a selection of system fonts into the registry.
pub fn load_cached_fonts() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        // App cache
        if let Some(dir) = fonts_dir() {
            if let Ok(rd) = std::fs::read_dir(&dir) {
                for ent in rd.flatten() {
                    let path = ent.path();
                    if is_font_path(&path) {
                        let name = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("Font")
                            .to_string();
                        try_register(&name, &path);
                    }
                }
            }
        }
        // System fonts (shallow + one level deep)
        scan_system_fonts();
    }
}

fn is_font_path(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    ext == "ttf" || ext == "otf"
}

#[cfg(not(target_arch = "wasm32"))]
fn scan_system_fonts() {
    for dir in system_font_dirs() {
        scan_font_dir(&dir, 0);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn scan_font_dir(dir: &Path, depth: u32) {
    if depth > 2 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for ent in rd.flatten() {
        let path = ent.path();
        if path.is_dir() {
            scan_font_dir(&path, depth + 1);
        } else if is_font_path(&path) {
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Font")
                .to_string();
            // Prefix so user can tell system vs downloaded
            let display = format!("System: {name}");
            try_register(&display, &path);
            // Also register bare name if free
            try_register(&name, &path);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn scan_system_fonts() {}

#[cfg(not(target_arch = "wasm32"))]
pub fn import_font_file(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Font")
        .to_string();
    {
        let mut reg = font_registry().lock().unwrap();
        reg.insert(name.clone(), bytes.clone())?;
    }
    if let Some(dir) = fonts_dir() {
        let dest = dir.join(
            path.file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("font.ttf")),
        );
        let _ = std::fs::write(dest, &bytes);
    }
    Ok(name)
}

pub fn is_catalog_installed(name: &str) -> bool {
    font_registry()
        .lock()
        .map(|r| r.contains(name))
        .unwrap_or(false)
}
