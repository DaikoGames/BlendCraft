//! Per-frame mesh sequences (the equivalent of Blender's "Mesh Sequence Cache").
//!
//! A `MeshSequence` stores one complete mesh per cache sample. The topology may change
//! from sample to sample, which is what liquid surfaces (fluid bakes and Alembic water
//! caches) need. The sequence is attached to an object; during timeline evaluation the
//! object's mesh is swapped for the sample that belongs to the current frame.
use crate::mesh::MeshData;
use crate::scene::Scene;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshSequence {
    /// Where the data came from ("fluid bake", an .abc path, ...). Informational.
    pub source: String,
    /// One mesh per cache sample.
    pub frames: Vec<Arc<MeshData>>,
    /// Scene frame (at the cache's own frame rate) of the first sample.
    pub start_frame: f32,
    /// Sample rate of the cache in samples per second.
    pub fps: f32,
    /// Blender's "Frame Offset": shifts the cache in time.
    pub frame_offset: f32,
    /// Blender's "Speed" multiplier (1 = real time).
    pub speed: f32,
    pub enabled: bool,
}

impl MeshSequence {
    pub fn new(source: impl Into<String>, frames: Vec<Arc<MeshData>>, start_frame: f32, fps: f32) -> Self {
        Self {
            source: source.into(),
            frames,
            start_frame,
            fps,
            frame_offset: 0.0,
            speed: 1.0,
            enabled: true,
        }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Sample index that is shown at `scene_frame`. The cache holds its first/last
    /// sample outside its range, like Blender does.
    pub fn index_at(&self, scene_frame: f32, scene_fps: f32) -> Option<usize> {
        if self.frames.is_empty() {
            return None;
        }
        let scene_fps = if scene_fps > 0.0 { scene_fps } else { 24.0 };
        let cache_fps = if self.fps > 0.0 { self.fps } else { scene_fps };
        let local = (scene_frame - self.frame_offset) * self.speed;
        let samples = (local - self.start_frame) * (cache_fps / scene_fps);
        let idx = (samples + 1e-3).floor();
        let last = (self.frames.len() - 1) as f32;
        Some(idx.clamp(0.0, last) as usize)
    }

    pub fn mesh_at(&self, scene_frame: f32, scene_fps: f32) -> Option<Arc<MeshData>> {
        let i = self.index_at(scene_frame, scene_fps)?;
        self.frames.get(i).cloned()
    }

    /// Last scene frame (rounded) that the cache covers.
    pub fn end_frame(&self, scene_fps: f32) -> f32 {
        let scene_fps = if scene_fps > 0.0 { scene_fps } else { 24.0 };
        let cache_fps = if self.fps > 0.0 { self.fps } else { scene_fps };
        let span = self.frames.len().saturating_sub(1) as f32 * (scene_fps / cache_fps);
        let speed = if self.speed.abs() > 1e-6 { self.speed } else { 1.0 };
        self.start_frame / speed + span / speed + self.frame_offset
    }
}

/// Swap every cached object's mesh for the sample belonging to `frame`.
/// Returns true when at least one mesh changed (the GPU scene must be rebuilt).
pub fn apply_mesh_caches(scene: &mut Scene, frame: f32) -> bool {
    if scene.mesh_caches.is_empty() {
        return false;
    }
    let fps = scene.animation.fps as f32;
    let mut updates: Vec<(u64, Arc<MeshData>)> = Vec::new();
    for (id, seq) in scene.mesh_caches.iter() {
        if !seq.enabled {
            continue;
        }
        if let Some(mesh) = seq.mesh_at(frame, fps) {
            updates.push((*id, mesh));
        }
    }
    let mut changed = false;
    for (id, mesh) in updates {
        if let Some(index) = scene.index_of(id) {
            let object = &mut scene.objects[index];
            if !Arc::ptr_eq(&object.mesh, &mesh) {
                object.mesh = mesh;
                changed = true;
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(n: usize) -> MeshSequence {
        let frames = (0..n).map(|_| Arc::new(MeshData::default())).collect();
        MeshSequence::new("test", frames, 1.0, 24.0)
    }

    #[test]
    fn index_follows_frame_and_clamps() {
        let s = seq(10);
        assert_eq!(s.index_at(1.0, 24.0), Some(0));
        assert_eq!(s.index_at(4.0, 24.0), Some(3));
        assert_eq!(s.index_at(-50.0, 24.0), Some(0));
        assert_eq!(s.index_at(500.0, 24.0), Some(9));
    }

    #[test]
    fn offset_and_cache_rate_are_respected() {
        let mut s = seq(10);
        s.frame_offset = 5.0;
        assert_eq!(s.index_at(6.0, 24.0), Some(0));
        let mut half = seq(10);
        half.fps = 12.0; // one sample per two scene frames
        assert_eq!(half.index_at(5.0, 24.0), Some(2));
    }
}
