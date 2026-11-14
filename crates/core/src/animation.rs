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

    pub fn insert_or_merge(
        &mut self,
        frame: i32,
        loc: Option<Vec3>,
        rot: Option<Quat>,
        scl: Option<Vec3>,
    ) {
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
            eval_vec3(
                self.keys
                    .iter()
                    .filter_map(|k| k.location.map(|v| (k.frame, v))),
                frame,
            ),
            eval_quat(
                self.keys
                    .iter()
                    .filter_map(|k| k.rotation.map(|v| (k.frame, v))),
                frame,
            ),
            eval_vec3(
                self.keys
                    .iter()
                    .filter_map(|k| k.scale.map(|v| (k.frame, v))),
                frame,
            ),
        )
    }
}

/// Monotone (PCHIP) tangent of one scalar channel at key `k`, in value-per-frame.
/// Never overshoots the keys and gives C1-continuous motion (no stop-and-go at
/// each key like per-segment ease-in/out would). With two keys it is exactly linear.
fn pchip_tangent(ts: &[f32], vs: &[f32], k: usize) -> f32 {
    let n = ts.len();
    let slope = |a: usize, b: usize| {
        let h = ts[b] - ts[a];
        if h.abs() < f32::EPSILON {
            0.0
        } else {
            (vs[b] - vs[a]) / h
        }
    };
    if k == 0 {
        return slope(0, 1);
    }
    if k == n - 1 {
        return slope(n - 2, n - 1);
    }
    let d0 = slope(k - 1, k);
    let d1 = slope(k, k + 1);
    if d0 * d1 <= 0.0 {
        return 0.0;
    }
    let h0 = ts[k] - ts[k - 1];
    let h1 = ts[k + 1] - ts[k];
    let w1 = 2.0 * h1 + h0;
    let w2 = h1 + 2.0 * h0;
    (w1 + w2) / (w1 / d0 + w2 / d1)
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
    let ts: Vec<f32> = keys.iter().map(|(f, _)| *f as f32).collect();
    for i in 0..keys.len() - 1 {
        let (f0, v0) = keys[i];
        let (f1, v1) = keys[i + 1];
        if frame >= f0 as f32 && frame <= f1 as f32 {
            let h = (f1 - f0) as f32;
            if h <= 0.0 {
                return Some(v1);
            }
            let t = (frame - f0 as f32) / h;
            let (t2, t3) = (t * t, t * t * t);
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;
            let mut out = [0.0f32; 3];
            for c in 0..3 {
                let vs: Vec<f32> = keys.iter().map(|(_, v)| v[c]).collect();
                let m0 = pchip_tangent(&ts, &vs, i);
                let m1 = pchip_tangent(&ts, &vs, i + 1);
                out[c] = h00 * v0[c] + h10 * h * m0 + h01 * v1[c] + h11 * h * m1;
            }
            return Some(Vec3::from_array(out));
        }
    }
    Some(keys[keys.len() - 1].1)
}

fn quat_log(q: Quat) -> Vec3 {
    let v = Vec3::new(q.x, q.y, q.z);
    let len = v.length();
    if len < 1e-6 {
        return Vec3::ZERO;
    }
    v / len * len.atan2(q.w)
}

fn quat_exp(u: Vec3) -> Quat {
    let theta = u.length();
    if theta < 1e-6 {
        return Quat::IDENTITY;
    }
    let a = u / theta * theta.sin();
    Quat::from_xyzw(a.x, a.y, a.z, theta.cos()).normalize()
}

/// Smooth rotation through the keys (squad). With two keys it is exactly slerp.
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
    // Keep neighbouring keys in the same hemisphere so we always take the short way.
    let mut qs: Vec<Quat> = Vec::with_capacity(keys.len());
    for (_, q) in &keys {
        let q = q.normalize();
        match qs.last() {
            Some(prev) if prev.dot(q) < 0.0 => qs.push(-q),
            _ => qs.push(q),
        }
    }
    // Inner control quaternions.
    let n = qs.len();
    let inner = |k: usize| -> Quat {
        if k == 0 || k == n - 1 {
            return qs[k];
        }
        let qi = qs[k];
        let inv = qi.conjugate();
        let l0 = quat_log((inv * qs[k - 1]).normalize());
        let l1 = quat_log((inv * qs[k + 1]).normalize());
        (qi * quat_exp(-(l0 + l1) * 0.25)).normalize()
    };
    for i in 0..n - 1 {
        let (f0, _) = keys[i];
        let (f1, _) = keys[i + 1];
        if frame >= f0 as f32 && frame <= f1 as f32 {
            let t = if f1 == f0 {
                0.0
            } else {
                (frame - f0 as f32) / (f1 - f0) as f32
            };
            let a = qs[i].slerp(qs[i + 1], t);
            let b = inner(i).slerp(inner(i + 1), t);
            return Some(a.slerp(b, 2.0 * t * (1.0 - t)).normalize());
        }
    }
    Some(qs[n - 1])
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
