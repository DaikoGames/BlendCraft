//! "Open Recent" list. On native: stored in a small text file in the user's config folder.
//! On wasm: in-memory only (no persistent storage for paths).
use std::path::{Path, PathBuf};

const MAX_RECENT: usize = 10;

#[cfg(not(target_arch = "wasm32"))]
fn config_file() -> Option<PathBuf> {
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)?
    } else if cfg!(target_os = "macos") {
        PathBuf::from(std::env::var_os("HOME")?).join("Library").join("Application Support")
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?
    };
    Some(base.join("BlendCraft").join("recent.txt"))
}

fn push_recent(list: &mut Vec<PathBuf>, p: PathBuf) {
    list.retain(|x| x != &p);
    list.insert(0, p);
    list.truncate(MAX_RECENT);
}

pub struct Recent {
    pub paths: Vec<PathBuf>,
}

impl Recent {
    pub fn load() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            return Self { paths: Vec::new() };
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut paths: Vec<PathBuf> = Vec::new();
            if let Some(text) = config_file().and_then(|f| std::fs::read_to_string(f).ok()) {
                for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
                    let p = PathBuf::from(line);
                    if p.exists() && !paths.contains(&p) {
                        paths.push(p);
                    }
                }
            }
            paths.truncate(MAX_RECENT);
            Self { paths }
        }
    }

    pub fn add(&mut self, p: &Path) {
        push_recent(&mut self.paths, p.to_path_buf());
        self.save();
    }

    pub fn remove(&mut self, p: &Path) {
        self.paths.retain(|x| x != p);
        self.save();
    }

    pub fn clear(&mut self) {
        self.paths.clear();
        self.save();
    }

    fn save(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(f) = config_file() else { return };
            if let Some(dir) = f.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let text: String = self.paths.iter().map(|p| format!("{}\n", p.to_string_lossy())).collect();
            if let Err(e) = std::fs::write(&f, text) {
                log::warn!("could not save recent files: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newest_first_no_duplicates_and_capped() {
        let mut l: Vec<PathBuf> = Vec::new();
        for i in 0..15 {
            push_recent(&mut l, PathBuf::from(format!("/p/{i}.forge")));
        }
        assert_eq!(l.len(), MAX_RECENT);
        assert_eq!(l[0], PathBuf::from("/p/14.forge"));
        push_recent(&mut l, PathBuf::from("/p/10.forge"));
        assert_eq!(l[0], PathBuf::from("/p/10.forge"));
        assert_eq!(l.iter().filter(|p| **p == PathBuf::from("/p/10.forge")).count(), 1);
    }
}
