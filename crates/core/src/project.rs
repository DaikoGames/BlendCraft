//! Native project format (.forge): magic + zlib(bincode(Scene)).
use crate::Scene;
use anyhow::{bail, Result};
use std::io::{Read, Write};

const MAGIC: &[u8; 4] = b"FRG3";

pub fn to_bytes(scene: &Scene) -> Result<Vec<u8>> {
    let raw = bincode::serialize(scene)?;
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    enc.write_all(&raw)?;
    let mut out = MAGIC.to_vec();
    out.extend(enc.finish()?);
    Ok(out)
}

pub fn from_bytes(bytes: &[u8]) -> Result<Scene> {
    if bytes.len() < 4 || &bytes[..4] != MAGIC {
        bail!("not a Forge3D project file");
    }
    let mut raw = Vec::new();
    flate2::read::ZlibDecoder::new(&bytes[4..]).read_to_end(&mut raw)?;
    Ok(bincode::deserialize(&raw)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;

    #[test]
    fn uv_coordinates_survive_project_roundtrip() {
        let mut d = Document::new();
        let obj = &mut d.scene.objects[0];
        let mesh = std::sync::Arc::make_mut(&mut obj.mesh);
        mesh.vertices[0].uv = [0.123, 0.875];
        let bytes = to_bytes(&d.scene).unwrap();
        let loaded = from_bytes(&bytes).unwrap();
        let uv = loaded.objects[0].mesh.vertices[0].uv;
        assert!((uv[0] - 0.123).abs() < 1e-6);
        assert!((uv[1] - 0.875).abs() < 1e-6);
    }

    #[test]
    fn active_edit_session_uvs_are_committed_before_project_save() {
        let mut d = Document::new();
        d.select_only(0);
        d.enter_edit_mode().unwrap();
        d.edit.as_mut().unwrap().mesh.vertices[0].uv = [0.314, 0.728];
        // This is the same commit performed by the application save command.
        d.sync_edit_mesh();
        let loaded = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        let uv = loaded.objects[0].mesh.vertices[0].uv;
        assert!((uv[0] - 0.314).abs() < 1e-6);
        assert!((uv[1] - 0.728).abs() < 1e-6);
    }

    #[test]
    fn painted_texture_survives_project_roundtrip() {
        let mut d = Document::new();
        d.select_only(0);
        d.uv_unwrap_smart();
        d.enter_paint_mode().unwrap();
        assert!(d.paint_at_uv(glam::Vec2::new(0.5, 0.5), [1.0, 0.0, 0.0, 1.0], 0.025, 1.0));
        let loaded = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        let material = &loaded.materials[loaded.objects[0].material.unwrap()];
        let tex = &loaded.textures[material.base_color_tex.unwrap()];
        let center = ((tex.height / 2 * tex.width + tex.width / 2) * 4) as usize;
        assert!(tex.rgba8[center] > 240);
        assert!(tex.rgba8[center + 1] < 20);
        assert!(tex.rgba8[center + 2] < 20);
    }

    #[test]
    fn roundtrip() {
        let d = Document::new();
        let back = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        assert_eq!(back.objects.len(), 3); // Cube + Sun + Camera
        assert_eq!(back.objects[0].mesh.vertices.len(), d.scene.objects[0].mesh.vertices.len());
    }
}
