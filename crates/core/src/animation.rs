//! Object transform keyframes and scene timeline (Blender-inspired).
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyChannel {
    Location,
    Rotation,
    Scale,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformKey {
    pub frame: i32,
    pub location: Option<Vec3>,
    pub rotation: Option<Quat>,
    pub scale: Option<Vec3>,
}

impl TransformKey {
    pub fn empty(frame: i32) -> Self {
        Self {
            frame,
            location: None,
            rotation: None,
            scale: None,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ObjectAnimation {
    pub keys: Vec<TransformKey>,
}

impl ObjectAnimation {
    pub fn sort(&mut self) {
        self.keys.sort_by_key(|k| k.frame);
    }

    pub fn insert_or_merge(&mut self, frame: i32, loc: Option<Vec3>, rot: Option<Quat>, scl: Option<Vec3>) {
        if let Some(k) = self.keys.iter_mut().find(|k| k.frame == frame) {
            if loc.is_some() {
                k.location = loc;
            }
            if rot.is_some() {
                k.rotation = rot;
            }
            if scl.is_some() {
                k.scale = scl;
            }
        } else {
            self.keys.push(TransformKey {
                frame,
                location: loc,
                rotation: rot,
                scale: scl,
            });
            self.sort();
        }
    }

    pub fn delete_at(&mut self, frame: i32) {
        self.keys.retain(|k| k.frame != frame);
    }

    pub fn has_any_at(&self, frame: i32) -> bool {
        self.keys.iter().any(|k| k.frame == frame)
    }

    /// Evaluate transform channels at `frame` (linear interpolation).
    pub fn eval(&self, frame: f32) -> (Option<Vec3>, Option<Quat>, Option<Vec3>) {
        if self.keys.is_empty() {
            return (None, None, None);
        }
        (
            eval_vec3(self.keys.iter().filter_map(|k| k.location.map(|v| (k.frame, v))), frame),
            eval_quat(self.keys.iter().filter_map(|k| k.rotation.map(|v| (k.frame, v))), frame),
            eval_vec3(self.keys.iter().filter_map(|k| k.scale.map(|v| (k.frame, v))), frame),
        )
    }
}

fn eval_vec3(keys: impl Iterator<Item = (i32, Vec3)>, frame: f32) -> Option<Vec3> {
    let mut keys: Vec<(i32, Vec3)> = keys.collect();
    if keys.is_empty() {
        return None;
    }
    keys.sort_by_key(|(f, _)| *f);
    if frame <= keys[0].0 as f32 {
        return Some(keys[0].1);
    }
    if frame >= keys[keys.len() - 1].0 as f32 {
        return Some(keys[keys.len() - 1].1);
    }
    for w in keys.windows(2) {
        let (f0, v0) = w[0];
        let (f1, v1) = w[1];
        if frame >= f0 as f32 && frame <= f1 as f32 {
            let t = if f1 == f0 {
                0.0
            } else {
                (frame - f0 as f32) / (f1 - f0) as f32
            };
            return Some(v0.lerp(v1, t));
        }
    }
    Some(keys[keys.len() - 1].1)
}

fn eval_quat(keys: impl Iterator<Item = (i32, Quat)>, frame: f32) -> Option<Quat> {
    let mut keys: Vec<(i32, Quat)> = keys.collect();
    if keys.is_empty() {
        return None;
    }
    keys.sort_by_key(|(f, _)| *f);
    if frame <= keys[0].0 as f32 {
        return Some(keys[0].1);
    }
    if frame >= keys[keys.len() - 1].0 as f32 {
        return Some(keys[keys.len() - 1].1);
    }
    for w in keys.windows(2) {
        let (f0, v0) = w[0];
        let (f1, v1) = w[1];
        if frame >= f0 as f32 && frame <= f1 as f32 {
            let t = if f1 == f0 {
                0.0
            } else {
                (frame - f0 as f32) / (f1 - f0) as f32
            };
            return Some(v0.slerp(v1, t));
        }
    }
    Some(keys[keys.len() - 1].1)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnimationData {
    pub frame_current: i32,
    pub frame_start: i32,
    pub frame_end: i32,
    pub fps: u32,
    /// object id → keys
    pub objects: HashMap<u64, ObjectAnimation>,
    #[serde(default)]
    pub playing: bool,
    #[serde(default)]
    pub auto_key: bool,
}

impl Default for AnimationData {
    fn default() -> Self {
        Self {
            frame_current: 1,
            frame_start: 1,
            frame_end: 250,
            fps: 24,
            objects: HashMap::new(),
            playing: false,
            auto_key: false,
        }
    }
}

impl AnimationData {
    pub fn set_frame(&mut self, f: i32) {
        self.frame_current = f.clamp(self.frame_start, self.frame_end);
    }

    pub fn step(&mut self) {
        if !self.playing {
            return;
        }
        self.frame_current += 1;
        if self.frame_current > self.frame_end {
            self.frame_current = self.frame_start;
        }
    }
}
