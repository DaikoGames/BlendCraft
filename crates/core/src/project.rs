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
    fn roundtrip() {
        let d = Document::new();
        let back = from_bytes(&to_bytes(&d.scene).unwrap()).unwrap();
        assert_eq!(back.objects.len(), 1);
        assert_eq!(back.objects[0].mesh.vertices.len(), d.scene.objects[0].mesh.vertices.len());
    }
}
