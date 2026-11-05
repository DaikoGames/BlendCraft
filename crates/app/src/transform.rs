//! Transform sessions (shared by the gizmo and the modal G/R/S operators), snapping and numeric input.
use crate::tools::{project_px, Orientation, Pivot, SnapMode, SnapSettings};
use forge_core::{Aabb, Document, OrbitCamera};
use glam::{Mat3, Mat4, Quat, Vec2, Vec3};
use std::collections::HashSet;
use std::f32::consts::{PI, TAU};
use winit::keyboard::KeyCode;

pub const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Move(Vec3),
    Rotate { axis: Vec3, angle: f32 },
    /// `axis: None` = uniform scale
    Scale { axis: Option<Vec3>, factor: f32 },
}

struct Item {
    id: u64,
    t: Vec3,
    r: Quat,
    s: Vec3,
    world: Mat4,
    parent_inv: Mat4,
}

/// Remembers the start state of the selected root objects and applies world-space operations to them.
pub struct Session {
    items: Vec<Item>,
    pub pivot: Vec3,
    individual: bool,
    pub exclude: HashSet<u64>,
}

pub fn pivot_point(doc: &Document, mode: Pivot) -> Option<Vec3> {
    let sel = doc.selected_indices();
    if sel.is_empty() {
        return None;
    }
    let w = doc.scene.world_matrices();
    let pos = |i: usize| w[i].w_axis.truncate();
    Some(match mode {
        Pivot::Cursor => doc.scene.cursor,
        Pivot::Active => doc.active_index().filter(|i| sel.contains(i)).map_or_else(|| pos(sel[0]), pos),
        Pivot::Median | Pivot::Individual => sel.iter().map(|&i| pos(i)).sum::<Vec3>() / sel.len() as f32,
        Pivot::BoundsCenter => sel
            .iter()
            .filter_map(|&i| doc.scene.object_world_bounds(i, &w[i]))
            .reduce(Aabb::union)
            .map_or(Vec3::ZERO, |b| b.center()),
    })
}

pub fn orientation(doc: &Document, cam: &OrbitCamera, mode: Orientation) -> Quat {
    match mode {
        Orientation::Global => Quat::IDENTITY,
        Orientation::Local => doc.active_index().map_or(Quat::IDENTITY, |a| {
            let (_, r, _) = doc.scene.world_matrices()[a].to_scale_rotation_translation();
            r
        }),
        Orientation::View => {
            let (_, r, _) = cam.view().inverse().to_scale_rotation_translation();
            r
        }
    }
}

impl Session {
    pub fn new(doc: &Document, pivot: Pivot) -> Option<Session> {
        let sel = doc.selected_indices();
        if sel.is_empty() {
            return None;
        }
        let worlds = doc.scene.world_matrices();
        let pv = pivot_point(doc, pivot)?;
        let items = doc
            .scene
            .roots(&sel)
            .into_iter()
            .map(|i| {
                let o = &doc.scene.objects[i];
                let pw = doc.scene.parent_world(i, &worlds);
                Item {
                    id: o.id,
                    t: o.translation,
                    r: o.rotation,
                    s: o.scale,
                    world: worlds[i],
                    parent_inv: if pw.determinant().abs() < 1e-12 { Mat4::IDENTITY } else { pw.inverse() },
                }
            })
            .collect();
        Some(Session { items, pivot: pv, individual: pivot == Pivot::Individual, exclude: doc.selection.clone().into_iter().collect() })
    }

    /// Apply `op` relative to the START state (so it can be called repeatedly while dragging).
    pub fn apply(&self, doc: &mut Document, op: Op) {
        for it in &self.items {
            let p = if self.individual { it.world.w_axis.truncate() } else { self.pivot };
            let around = |m: Mat4| Mat4::from_translation(p) * m * Mat4::from_translation(-p);
            let t = match op {
                Op::Move(d) => Mat4::from_translation(d),
                Op::Rotate { axis, angle } => around(Mat4::from_quat(Quat::from_axis_angle(axis.normalize_or_zero(), angle))),
                Op::Scale { axis: Some(a), factor } => {
                    let a = a.normalize_or_zero();
                    let outer = Mat3::from_cols(a * a.x, a * a.y, a * a.z);
                    around(Mat4::from_mat3(Mat3::IDENTITY + outer * (factor - 1.0)))
                }
                Op::Scale { axis: None, factor } => around(Mat4::from_scale(Vec3::splat(factor))),
            };
            let local = it.parent_inv * (t * it.world);
            let (s, r, tr) = local.to_scale_rotation_translation();
            if let Some(i) = doc.scene.index_of(it.id) {
                let o = &mut doc.scene.objects[i];
                o.scale = s;
                o.rotation = r;
                o.translation = tr;
            }
        }
        doc.touch();
    }

    pub fn restore(&self, doc: &mut Document) {
        for it in &self.items {
            if let Some(i) = doc.scene.index_of(it.id) {
                let o = &mut doc.scene.objects[i];
                o.translation = it.t;
                o.rotation = it.r;
                o.scale = it.s;
            }
        }
        doc.touch();
    }
}

// ---------------------------------------------------------------- geometry helpers
pub fn cursor_ray(cursor: (f64, f64), cam: &OrbitCamera, vp: [f32; 4]) -> (Vec3, Vec3) {
    let nx = (cursor.0 as f32 - vp[0]) / vp[2] * 2.0 - 1.0;
    let ny = 1.0 - (cursor.1 as f32 - vp[1]) / vp[3] * 2.0;
    cam.ray((nx, ny), vp[2] / vp[3])
}

/// Parameter t on the line `lo + t*ld` that is closest to the ray `ro + s*rd`.
pub fn closest_param(ro: Vec3, rd: Vec3, lo: Vec3, ld: Vec3) -> Option<f32> {
    let w0 = ro - lo;
    let (a, b, c) = (rd.dot(rd), rd.dot(ld), ld.dot(ld));
    let (d, e) = (rd.dot(w0), ld.dot(w0));
    let denom = a * c - b * b;
    if denom.abs() < 1e-8 {
        return None;
    }
    Some((a * e - b * d) / denom)
}

pub fn ray_plane(ro: Vec3, rd: Vec3, point: Vec3, normal: Vec3) -> Option<Vec3> {
    let denom = rd.dot(normal);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (point - ro).dot(normal) / denom;
    Some(ro + rd * t)
}

/// Nearest vertex (of objects not in `exclude`) to the cursor, within 20 px.
pub fn vertex_snap(doc: &Document, cam: &OrbitCamera, vp: [f32; 4], cursor: (f64, f64), exclude: &HashSet<u64>) -> Option<Vec3> {
    let worlds = doc.scene.world_matrices();
    let m = cam.view_proj(vp[2] / vp[3]);
    let cur = Vec2::new(cursor.0 as f32, cursor.1 as f32);
    let mut best: Option<(f32, Vec3)> = None;
    for (i, o) in doc.scene.objects.iter().enumerate() {
        if !doc.scene.is_visible(i) || exclude.contains(&o.id) || !o.kind.is_mesh() {
            continue;
        }
        let w = worlds[i];
        let step = (o.mesh.vertices.len() / 50_000).max(1); // keep huge meshes responsive
        for v in o.mesh.vertices.iter().step_by(step) {
            let p = w.transform_point3(Vec3::from(v.position));
            let c = m * p.extend(1.0);
            if c.w <= 1e-4 {
                continue;
            }
            let s = Vec2::new(vp[0] + (c.x / c.w * 0.5 + 0.5) * vp[2], vp[1] + (1.0 - (c.y / c.w * 0.5 + 0.5)) * vp[3]);
            let d = (s - cur).length();
            if d < 20.0 && best.map_or(true, |(bd, _)| d < bd) {
                best = Some((d, p));
            }
        }
    }
    best.map(|b| b.1)
}

// ---------------------------------------------------------------- modal G / R / S
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModalKind {
    Move,
    Rotate,
    Scale,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Constrain {
    Free,
    /// axis index, local?
    Axis(usize, bool),
    Plane(usize, bool),
}

pub struct Modal {
    pub kind: ModalKind,
    session: Session,
    start: (f64, f64),
    constrain: Constrain,
    numeric: String,
    took_checkpoint: bool,
    local_rot: Quat,
    last_angle: f32,
    accum: f32,
    info: String,
}

impl Modal {
    pub fn start(kind: ModalKind, doc: &mut Document, cam: &OrbitCamera, vp: [f32; 4], cursor: (f64, f64), pivot: Pivot, checkpoint: bool) -> Option<Modal> {
        let session = Session::new(doc, pivot)?;
        if checkpoint {
            doc.checkpoint();
        }
        let last_angle = project_px(cam, vp, session.pivot)
            .map(|ps| {
                let c = Vec2::new(cursor.0 as f32, cursor.1 as f32) - ps;
                c.y.atan2(c.x)
            })
            .unwrap_or(0.0);
        Some(Modal {
            kind,
            session,
            start: cursor,
            constrain: Constrain::Free,
            numeric: String::new(),
            took_checkpoint: checkpoint,
            local_rot: orientation(doc, cam, Orientation::Local),
            last_angle,
            accum: 0.0,
            info: String::new(),
        })
    }

    fn dir(&self, a: usize, local: bool) -> Vec3 {
        if local { self.local_rot * AXES[a] } else { AXES[a] }
    }

    pub fn status(&self) -> String {
        self.info.clone()
    }

    /// Returns true if the key was used by the modal operator.
    pub fn key(&mut self, code: KeyCode, shift: bool) -> bool {
        use KeyCode as K;
        let axis = match code {
            K::KeyX => Some(0),
            K::KeyY => Some(1),
            K::KeyZ => Some(2),
            _ => None,
        };
        if let Some(a) = axis {
            self.constrain = if shift && self.kind == ModalKind::Move {
                match self.constrain {
                    Constrain::Plane(x, false) if x == a => Constrain::Plane(a, true),
                    Constrain::Plane(x, true) if x == a => Constrain::Free,
                    _ => Constrain::Plane(a, false),
                }
            } else {
                match self.constrain {
                    Constrain::Axis(x, false) if x == a => Constrain::Axis(a, true),
                    Constrain::Axis(x, true) if x == a => Constrain::Free,
                    _ => Constrain::Axis(a, false),
                }
            };
            return true;
        }
        let digit = match code {
            K::Digit0 | K::Numpad0 => Some('0'),
            K::Digit1 | K::Numpad1 => Some('1'),
            K::Digit2 | K::Numpad2 => Some('2'),
            K::Digit3 | K::Numpad3 => Some('3'),
            K::Digit4 | K::Numpad4 => Some('4'),
            K::Digit5 | K::Numpad5 => Some('5'),
            K::Digit6 | K::Numpad6 => Some('6'),
            K::Digit7 | K::Numpad7 => Some('7'),
            K::Digit8 | K::Numpad8 => Some('8'),
            K::Digit9 | K::Numpad9 => Some('9'),
            _ => None,
        };
        if let Some(c) = digit {
            self.numeric.push(c);
            return true;
        }
        match code {
            K::Period | K::NumpadDecimal => {
                if !self.numeric.contains('.') {
                    if self.numeric.is_empty() || self.numeric == "-" {
                        self.numeric.push('0');
                    }
                    self.numeric.push('.');
                }
                true
            }
            K::Minus | K::NumpadSubtract => {
                if self.numeric.starts_with('-') {
                    self.numeric.remove(0);
                } else {
                    self.numeric.insert(0, '-');
                }
                true
            }
            K::Backspace => {
                self.numeric.pop();
                true
            }
            _ => false,
        }
    }

    fn constraint_label(&self) -> String {
        let axis = |a: usize| ["X", "Y", "Z"][a];
        match self.constrain {
            Constrain::Free => String::new(),
            Constrain::Axis(a, l) => format!("  [{}{}]", if l { "local " } else { "" }, axis(a)),
            Constrain::Plane(a, l) => format!("  [plane, not {}{}]", if l { "local " } else { "" }, axis(a)),
        }
    }

    pub fn update(&mut self, doc: &mut Document, cam: &OrbitCamera, vp: [f32; 4], cursor: (f64, f64), snap: &SnapSettings, ctrl: bool) {
        let snapping = snap.enabled ^ ctrl;
        let num = self.numeric.parse::<f32>().ok();
        let (ro, rd) = cursor_ray(cursor, cam, vp);
        let (r0, d0) = cursor_ray(self.start, cam, vp);
        let back = cam.view().inverse().z_axis.truncate().normalize_or_zero();
        let pivot = self.session.pivot;
        let cur2 = Vec2::new(cursor.0 as f32, cursor.1 as f32);
        let start2 = Vec2::new(self.start.0 as f32, self.start.1 as f32);

        let op = match self.kind {
            ModalKind::Move => {
                let mut d = Vec3::ZERO;
                if let Some(v) = num {
                    d = match self.constrain {
                        Constrain::Axis(a, l) => self.dir(a, l) * v,
                        _ => Vec3::X * v,
                    };
                } else {
                    match self.constrain {
                        Constrain::Free => {
                            if let (Some(p0), Some(p1)) = (ray_plane(r0, d0, pivot, back), ray_plane(ro, rd, pivot, back)) {
                                d = p1 - p0;
                            }
                        }
                        Constrain::Axis(a, l) => {
                            let ax = self.dir(a, l);
                            if let (Some(t0), Some(t1)) = (closest_param(r0, d0, pivot, ax), closest_param(ro, rd, pivot, ax)) {
                                d = ax * (t1 - t0);
                            }
                        }
                        Constrain::Plane(a, l) => {
                            let n = self.dir(a, l);
                            if let (Some(p0), Some(p1)) = (ray_plane(r0, d0, pivot, n), ray_plane(ro, rd, pivot, n)) {
                                d = p1 - p0;
                            }
                        }
                    }
                    if snapping {
                        match snap.mode {
                            SnapMode::Increment => {
                                let st = snap.move_step.max(1e-4);
                                d = match self.constrain {
                                    Constrain::Axis(a, l) => {
                                        let ax = self.dir(a, l);
                                        ax * ((d.dot(ax) / st).round() * st)
                                    }
                                    _ => Vec3::new((d.x / st).round() * st, (d.y / st).round() * st, (d.z / st).round() * st),
                                };
                            }
                            SnapMode::Vertex => {
                                if let Some(p) = vertex_snap(doc, cam, vp, cursor, &self.session.exclude) {
                                    let mut dd = p - pivot;
                                    if let Constrain::Axis(a, l) = self.constrain {
                                        let ax = self.dir(a, l);
                                        dd = ax * dd.dot(ax);
                                    }
                                    d = dd;
                                }
                            }
                        }
                    }
                }
                self.info = format!("Move  D: {:.3}  {:.3}  {:.3}{}{}", d.x, d.y, d.z, self.constraint_label(), self.numeric_label());
                Op::Move(d)
            }
            ModalKind::Rotate => {
                let axis = match self.constrain {
                    Constrain::Axis(a, l) => self.dir(a, l),
                    _ => back,
                };
                let angle = if let Some(v) = num {
                    v.to_radians()
                } else {
                    if let Some(ps) = project_px(cam, vp, pivot) {
                        let c = cur2 - ps;
                        let a1 = c.y.atan2(c.x);
                        let mut dlt = a1 - self.last_angle;
                        while dlt > PI {
                            dlt -= TAU;
                        }
                        while dlt < -PI {
                            dlt += TAU;
                        }
                        self.accum += dlt;
                        self.last_angle = a1;
                    }
                    let mut a = if axis.dot(back) >= 0.0 { -self.accum } else { self.accum };
                    if snapping {
                        let st = snap.rotate_step_deg.to_radians().max(1e-3);
                        a = (a / st).round() * st;
                    }
                    a
                };
                self.info = format!("Rotate  {:.1}°{}{}", angle.to_degrees(), self.constraint_label(), self.numeric_label());
                Op::Rotate { axis, angle }
            }
            ModalKind::Scale => {
                let mut f = if let Some(v) = num {
                    v
                } else {
                    match project_px(cam, vp, pivot) {
                        Some(ps) => (cur2 - ps).length() / (start2 - ps).length().max(1.0),
                        None => 1.0,
                    }
                };
                if snapping && num.is_none() {
                    let st = snap.scale_step.max(1e-3);
                    f = (f / st).round() * st;
                }
                let axis = match self.constrain {
                    Constrain::Axis(a, l) => Some(self.dir(a, l)),
                    _ => None,
                };
                self.info = format!("Scale  {:.3}{}{}", f, self.constraint_label(), self.numeric_label());
                Op::Scale { axis, factor: f }
            }
        };
        self.session.apply(doc, op);
    }

    fn numeric_label(&self) -> String {
        if self.numeric.is_empty() { String::new() } else { format!("  = {}", self.numeric) }
    }

    pub fn confirm(self, doc: &mut Document) {
        doc.touch();
    }

    pub fn cancel(self, doc: &mut Document) {
        self.session.restore(doc);
        if self.took_checkpoint {
            doc.drop_last_checkpoint();
        }
    }
}
