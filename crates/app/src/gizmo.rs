//! Transform gizmo: Move (axis arrows), Rotate (axis rings), Scale (axis handles).
//! Works on the whole selection, honours pivot point, orientation and snapping.
use crate::tools::{project_px, Pivot, SnapMode, Tool, ToolState};
use crate::transform::{closest_param, cursor_ray, orientation, pivot_point, ray_plane, vertex_snap, Op, Session, AXES};
use egui::{Color32, Stroke};
use forge_core::{Document, OrbitCamera};
use glam::{Quat, Vec2, Vec3};
use std::collections::{BTreeSet, HashSet};
use std::f32::consts::TAU;

struct BoneItem {
    index: u32,
    head: [f32; 3],
    tail: [f32; 3],
    pose_rot: [f32; 4],
    pose_loc: [f32; 3],
    pose_scale: [f32; 3],
}

struct BoneSession {
    items: Vec<BoneItem>,
    /// Rest endpoints captured when the gizmo drag begins. These are used to
    /// identify actual connected joints consistently for every drag update.
    rest_joints: Vec<(Vec3, Vec3)>,
    inv_world: glam::Mat4,
    pivot_local: Vec3,
    pose_mode: bool,
}

enum TransformSession {
    Objects(Session),
    Bones(BoneSession),
}

impl BoneSession {
    fn new(doc: &Document, pivot: Vec3) -> Option<Self> {
        let id = doc.active?;
        let index = doc.scene.index_of(id)?;
        let world = doc.scene.world_matrices()[index];
        let inv_world = world.inverse();
        let arm = doc.active_armature()?;
        let mut indices: Vec<u32> = arm.selected.iter().copied().collect();
        if indices.is_empty() { if let Some(active) = arm.active { indices.push(active); } }
        if indices.is_empty() { return None; }
        let items = indices.into_iter().filter_map(|i| arm.bones.get(i as usize).map(|b| BoneItem {
            index: i, head: b.head, tail: b.tail, pose_rot: b.pose_rot,
            pose_loc: b.pose_loc, pose_scale: b.pose_scale,
        })).collect::<Vec<_>>();
        if items.is_empty() { return None; }
        let rest_joints = arm.bones.iter()
            .map(|bone| (bone.head_v(), bone.tail_v()))
            .collect();
        Some(Self {
            items,
            rest_joints,
            inv_world,
            pivot_local: inv_world.transform_point3(pivot),
            pose_mode: doc.pose_mode,
        })
    }

    fn apply(&self, doc: &mut Document, op: Op) {
        let local_axis = |axis: Vec3| self.inv_world.transform_vector3(axis).normalize_or_zero();
        if let Some(arm) = doc.active_armature_mut() {
            for item in &self.items {
                let Some(bone) = arm.bones.get_mut(item.index as usize) else { continue };
                let rest_head = Vec3::from(item.head);
                let rest_tail = Vec3::from(item.tail);
                let rest_dir = (rest_tail - rest_head).normalize_or_zero();
                let rest_rot = if rest_dir.length_squared() < 1e-10 { Quat::IDENTITY } else { Quat::from_rotation_arc(Vec3::Y, rest_dir) };
                match op {
                    Op::Move(delta) => {
                        let d = self.inv_world.transform_vector3(delta);
                        if self.pose_mode {
                            // pose_loc is stored in armature-local coordinates.
                            bone.pose_loc = (Vec3::from(item.pose_loc) + d).to_array();
                        } else {
                            bone.head = (rest_head + d).to_array();
                            bone.tail = (rest_tail + d).to_array();
                        }
                    }
                    Op::Rotate { axis, angle } => {
                        let axis_local = local_axis(axis);
                        let q = Quat::from_axis_angle(axis_local, angle);
                        if self.pose_mode {
                            let posed_head = rest_head + Vec3::from(item.pose_loc);
                            let new_head = self.pivot_local + q * (posed_head - self.pivot_local);
                            bone.pose_loc = (new_head - rest_head).to_array();
                            let bone_axis = rest_rot.conjugate() * axis_local;
                            let delta = Quat::from_axis_angle(bone_axis.normalize_or_zero(), angle);
                            let original = Quat::from_xyzw(item.pose_rot[0], item.pose_rot[1], item.pose_rot[2], item.pose_rot[3]).normalize();
                            let result = (delta * original).normalize();
                            bone.pose_rot = [result.x, result.y, result.z, result.w];
                        } else {
                            bone.head = (self.pivot_local + q * (rest_head - self.pivot_local)).to_array();
                            bone.tail = (self.pivot_local + q * (rest_tail - self.pivot_local)).to_array();
                        }
                    }
                    Op::Scale { axis, factor } => {
                        let scale_point = |p: Vec3| {
                            let d = p - self.pivot_local;
                            let scaled = if let Some(a) = axis {
                                let a = local_axis(a);
                                d + a * (d.dot(a) * (factor - 1.0))
                            } else { d * factor };
                            self.pivot_local + scaled
                        };
                        if self.pose_mode {
                            let posed_head = rest_head + Vec3::from(item.pose_loc);
                            bone.pose_loc = (scale_point(posed_head) - rest_head).to_array();
                            let mut scale = Vec3::from(item.pose_scale);
                            if let Some(a) = axis {
                                let a = local_axis(a);
                                let pose_q = Quat::from_xyzw(item.pose_rot[0], item.pose_rot[1], item.pose_rot[2], item.pose_rot[3]).normalize();
                                let posed_basis = rest_rot * pose_q;
                                let x = posed_basis * Vec3::X;
                                let y = posed_basis * Vec3::Y;
                                let z = posed_basis * Vec3::Z;
                                scale *= Vec3::new(
                                    1.0 + (factor - 1.0) * x.dot(a).powi(2),
                                    1.0 + (factor - 1.0) * y.dot(a).powi(2),
                                    1.0 + (factor - 1.0) * z.dot(a).powi(2),
                                );
                            } else { scale *= Vec3::splat(factor); }
                            bone.pose_scale = scale.clamp(Vec3::splat(0.01), Vec3::splat(100.0)).to_array();
                        } else {
                            bone.head = scale_point(rest_head).to_array();
                            bone.tail = scale_point(rest_tail).to_array();
                        }
                    }
                }
            }
            // The toolbar gizmo is a separate transform path from the modal
            // G/R/S tools in main.rs. Keep connected rest-pose joints in sync
            // here as well, otherwise dragging a gizmo silently bypasses the
            // armature connection rules.
            if !self.pose_mode {
                let selected: BTreeSet<u32> = self.items.iter().map(|item| item.index).collect();
                arm.propagate_connected_rest_joints(&self.rest_joints, &selected);
            }
        }
        if self.pose_mode { doc.apply_all_skinning(); }
        doc.touch();
    }
}

struct Drag {
    axis: usize,
    axis_dir: Vec3,
    orient: Quat,
    origin: Vec3,
    /// Move/Scale: start parameter along the axis. Rotate: start angle.
    start: f32,
    start_point: Option<Vec3>,
    tool: Tool,
    session: TransformSession,
    exclude: HashSet<u64>,
}

pub struct Gizmo {
    hover: Option<usize>,
    drag: Option<Drag>,
}

const COLORS: [Color32; 3] = [Color32::from_rgb(230, 60, 60), Color32::from_rgb(80, 200, 80), Color32::from_rgb(70, 120, 240)];
const ACTIVE: Color32 = Color32::from_rgb(255, 220, 60);
const GRAB_PX: f32 = 10.0;

fn ring_basis(a: usize, orient: Quat) -> (Vec3, Vec3) {
    let (u, v) = match a {
        0 => (Vec3::Y, Vec3::Z),
        1 => (Vec3::Z, Vec3::X),
        _ => (Vec3::X, Vec3::Y),
    };
    (orient * u, orient * v)
}

fn plane_angle(ro: Vec3, rd: Vec3, origin: Vec3, axis_dir: Vec3, orient: Quat, axis: usize) -> Option<f32> {
    let denom = rd.dot(axis_dir);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (origin - ro).dot(axis_dir) / denom;
    let d = ro + rd * t - origin;
    let (u, v) = ring_basis(axis, orient);
    Some(d.dot(v).atan2(d.dot(u)))
}

fn seg_dist(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let l2 = ab.length_squared();
    let t = if l2 < 1e-6 { 0.0 } else { ((p - a).dot(ab) / l2).clamp(0.0, 1.0) };
    (p - (a + ab * t)).length()
}

impl Gizmo {
    pub fn new() -> Self {
        Self { hover: None, drag: None }
    }

    pub fn dragging(&self) -> bool {
        self.drag.is_some()
    }

    pub fn end(&mut self) {
        self.drag = None;
    }

    /// (origin, axis orientation, handle length in world units)
    fn bone_pivot(doc: &Document, pivot: Pivot) -> Option<Vec3> {
        let id = doc.active?;
        let index = doc.scene.index_of(id)?;
        let world = doc.scene.world_matrices()[index];
        let arm = doc.active_armature()?;
        let mut selected: Vec<u32> = arm.selected.iter().copied().collect();
        if selected.is_empty() { if let Some(active) = arm.active { selected.push(active); } }
        if selected.is_empty() { return None; }
        let lines = arm.indexed_lines(doc.pose_mode);
        let mut points: Vec<(u32, Vec3, Vec3)> = Vec::new();
        for bone_index in selected.iter().copied() {
            let segments: Vec<_> = lines.iter().filter(|(i, _, _)| *i == bone_index).collect();
            if let (Some(first), Some(second)) = (segments.first(), segments.get(1)) {
                points.push((bone_index, world.transform_point3(first.1), world.transform_point3(second.2)));
            }
        }
        if points.is_empty() { return None; }
        let origin = match pivot {
            Pivot::Cursor => doc.scene.cursor,
            Pivot::Active | Pivot::Individual => {
                let active = arm.active.unwrap_or(selected[0]);
                let (_, h, t) = points.iter().find(|(i, _, _)| *i == active).or_else(|| points.first())?;
                (*h + *t) * 0.5
            }
            Pivot::Median => points.iter().map(|(_, h, t)| (*h + *t) * 0.5).sum::<Vec3>() / points.len() as f32,
            Pivot::BoundsCenter => {
                let mut lo = Vec3::splat(f32::INFINITY);
                let mut hi = Vec3::splat(f32::NEG_INFINITY);
                for (_, h, t) in &points { lo = lo.min(*h).min(*t); hi = hi.max(*h).max(*t); }
                (lo + hi) * 0.5
            }
        };
        Some(origin)
    }

    fn frame(&self, ts: &ToolState, cam: &OrbitCamera, doc: &Document) -> Option<(Vec3, Quat, f32)> {
        if !ts.tool.is_gizmo() { return None; }
        let origin = if doc.armature_edit || doc.pose_mode {
            Self::bone_pivot(doc, ts.pivot)?
        } else {
            let mode = if ts.pivot == Pivot::Individual { Pivot::Active } else { ts.pivot };
            pivot_point(doc, mode)?
        };
        let orient = orientation(doc, cam, ts.orient);
        let len = (cam.eye() - origin).length().max(0.01) * 0.15;
        Some((origin, orient, len))
    }

    fn hit_axis(&self, tool: Tool, cursor: (f64, f64), cam: &OrbitCamera, vp: [f32; 4], origin: Vec3, orient: Quat, len: f32) -> Option<usize> {
        let c = Vec2::new(cursor.0 as f32, cursor.1 as f32);
        if tool == Tool::Scale {
            if let Some(center) = project_px(cam, vp, origin) {
                let distance = (c - center).length();
                if distance <= 8.0 { return Some(3); }
            }
        }
        if tool == Tool::Move {
            let plane_axes = [(0usize, 1usize), (1, 2), (0, 2)];
            for (plane, (a, b)) in plane_axes.into_iter().enumerate() {
                let center_world = origin + (orient * AXES[a] + orient * AXES[b]) * (len * 0.18);
                if let Some(center) = project_px(cam, vp, center_world) {
                    if (c - center).length() <= 7.0 { return Some(3 + plane); }
                }
            }
        }
        let mut best: Option<(usize, f32)> = None;
        for a in 0..3 {
            let dir = orient * AXES[a];
            let d = match tool {
                Tool::Move | Tool::Scale => {
                    let (Some(p0), Some(p1)) = (project_px(cam, vp, origin), project_px(cam, vp, origin + dir * len)) else { continue };
                    seg_dist(c, p0, p1)
                }
                Tool::Rotate => {
                    let (u, v) = ring_basis(a, orient);
                    let mut m = f32::MAX;
                    let mut prev: Option<Vec2> = None;
                    for k in 0..=64 {
                        let th = k as f32 / 64.0 * TAU;
                        let p = origin + len * (th.cos() * u + th.sin() * v);
                        let Some(s) = project_px(cam, vp, p) else {
                            prev = None;
                            continue;
                        };
                        if let Some(pp) = prev {
                            m = m.min(seg_dist(c, pp, s));
                        }
                        prev = Some(s);
                    }
                    m
                }
                _ => continue,
            };
            if d < GRAB_PX && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((a, d));
            }
        }
        best.map(|b| b.0)
    }

    pub fn hover_test(&mut self, cursor: (f64, f64), cam: &OrbitCamera, vp: [f32; 4], doc: &Document, ts: &ToolState) {
        self.hover = self.frame(ts, cam, doc).and_then(|(origin, orient, len)| self.hit_axis(ts.tool, cursor, cam, vp, origin, orient, len));
    }

    /// Start dragging a handle if the cursor is on one. Takes an undo snapshot.
    pub fn begin(&mut self, cursor: (f64, f64), cam: &OrbitCamera, vp: [f32; 4], doc: &mut Document, ts: &ToolState) -> bool {
        let Some((origin, orient, len)) = self.frame(ts, cam, doc) else { return false };
        let Some(axis) = self.hit_axis(ts.tool, cursor, cam, vp, origin, orient, len) else { return false };
        let axis_dir = if axis < 3 { orient * AXES[axis] } else if ts.tool == Tool::Move {
            let normal_axis = match axis { 3 => 2, 4 => 0, _ => 1 };
            orient * AXES[normal_axis]
        } else {
            cam.view().inverse().z_axis.truncate().normalize_or_zero()
        };
        let (ro, rd) = cursor_ray(cursor, cam, vp);
        let start_point = if ts.tool == Tool::Move && axis >= 3 {
            ray_plane(ro, rd, origin, axis_dir)
        } else { None };
        let start = if ts.tool == Tool::Move && axis >= 3 {
            Some(0.0)
        } else if ts.tool == Tool::Scale && axis == 3 {
            project_px(cam, vp, origin).map(|center| (Vec2::new(cursor.0 as f32, cursor.1 as f32) - center).length())
        } else {
            match ts.tool {
                Tool::Rotate => plane_angle(ro, rd, origin, axis_dir, orient, axis),
                _ => closest_param(ro, rd, origin, axis_dir),
            }
        };
        let Some(start) = start else { return false };
        if ts.tool == Tool::Move && axis >= 3 && start_point.is_none() { return false; }
        let (session, exclude) = if doc.armature_edit || doc.pose_mode {
            let Some(session) = BoneSession::new(doc, origin) else { return false };
            (TransformSession::Bones(session), HashSet::new())
        } else {
            let Some(session) = Session::new(doc, ts.pivot) else { return false };
            let exclude = session.exclude.clone();
            (TransformSession::Objects(session), exclude)
        };
        doc.checkpoint();
        self.drag = Some(Drag { axis, axis_dir, orient, origin, start, start_point, tool: ts.tool, session, exclude });
        true
    }

    pub fn update(&mut self, cursor: (f64, f64), cam: &OrbitCamera, vp: [f32; 4], doc: &mut Document, ts: &ToolState, ctrl: bool) {
        let Some(d) = self.drag.as_ref() else { return };
        let snapping = ts.snap.enabled ^ ctrl;
        let (ro, rd) = cursor_ray(cursor, cam, vp);
        let op = match d.tool {
            Tool::Move => {
                let mut delta = if d.axis >= 3 {
                    let Some(point) = ray_plane(ro, rd, d.origin, d.axis_dir) else { return };
                    point - d.start_point.unwrap_or(d.origin)
                } else {
                    let Some(t) = closest_param(ro, rd, d.origin, d.axis_dir) else { return };
                    d.axis_dir * (t - d.start)
                };
                if snapping {
                    match ts.snap.mode {
                        SnapMode::Increment => {
                            let st = ts.snap.move_step.max(1e-4);
                            delta = Vec3::new((delta.x / st).round() * st, (delta.y / st).round() * st, (delta.z / st).round() * st);
                        }
                        SnapMode::Vertex => {
                            if let Some(p) = vertex_snap(doc, cam, vp, cursor, &d.exclude) {
                                delta = p - d.origin;
                                if d.axis < 3 {
                                    delta = d.axis_dir * delta.dot(d.axis_dir);
                                } else {
                                    delta -= d.axis_dir * delta.dot(d.axis_dir);
                                }
                            }
                        }
                    }
                }
                Op::Move(delta)
            }
            Tool::Rotate => {
                let Some(a) = plane_angle(ro, rd, d.origin, d.axis_dir, d.orient, d.axis) else { return };
                let mut angle = a - d.start;
                if snapping {
                    let st = ts.snap.rotate_step_deg.to_radians().max(1e-3);
                    angle = (angle / st).round() * st;
                }
                Op::Rotate { axis: d.axis_dir, angle }
            }
            Tool::Scale => {
                let t = if d.axis == 3 {
                    let Some(center) = project_px(cam, vp, d.origin) else { return };
                    (Vec2::new(cursor.0 as f32, cursor.1 as f32) - center).length()
                } else {
                    let Some(value) = closest_param(ro, rd, d.origin, d.axis_dir) else { return };
                    value
                };
                let mut f = if d.axis == 3 {
                    ((t - d.start) * 0.01).exp().clamp(0.01, 100.0)
                } else {
                    if d.start.abs() < 1e-4 { return; }
                    (t / d.start).max(0.01)
                };
                if snapping {
                    let st = ts.snap.scale_step.max(1e-3);
                    f = ((f / st).round() * st).max(st);
                }
                Op::Scale { axis: if d.axis == 3 { None } else { Some(d.axis_dir) }, factor: f }
            }
            _ => return,
        };
        match &d.session {
            TransformSession::Objects(session) => session.apply(doc, op),
            TransformSession::Bones(session) => session.apply(doc, op),
        }
    }

    /// Draws the handles. `rect` is the 3D viewport in egui points.
    pub fn draw(&self, ctx: &egui::Context, cam: &OrbitCamera, rect: egui::Rect, doc: &Document, ts: &ToolState) {
        let Some((origin, orient, len)) = self.frame(ts, cam, doc) else { return };
        let ppp = ctx.pixels_per_point();
        let vp = [rect.min.x * ppp, rect.min.y * ppp, rect.width() * ppp, rect.height() * ppp];
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("gizmo"))).with_clip_rect(rect);
        let pt = |p: Vec2| egui::pos2(p.x / ppp, p.y / ppp);
        let active = self.drag.as_ref().map(|d| d.axis).or(self.hover);
        if ts.tool == Tool::Scale {
            if let Some(center) = project_px(cam, vp, origin) {
                let color = if active == Some(3) { ACTIVE } else { Color32::from_gray(220) };
                painter.circle_filled(pt(center), 5.5, Color32::from_black_alpha(220));
                painter.circle_filled(pt(center), 3.5, color);
            }
        }
        if ts.tool == Tool::Move {
            for (plane, (a, b)) in [(0usize, 1usize), (1, 2), (0, 2)].into_iter().enumerate() {
                let center_world = origin + (orient * AXES[a] + orient * AXES[b]) * (len * 0.18);
                if let Some(center) = project_px(cam, vp, center_world) {
                    let color = if active == Some(3 + plane) { ACTIVE } else { Color32::from_rgba_unmultiplied(210, 210, 220, 190) };
                    painter.rect_filled(egui::Rect::from_center_size(pt(center), egui::vec2(9.0, 9.0)), 1.5, color);
                }
            }
        }

        for a in 0..3 {
            let on = active == Some(a);
            let col = if on { ACTIVE } else { COLORS[a] };
            let stroke = Stroke::new(if on { 3.5_f32 } else { 2.5_f32 }, col);
            let dir = orient * AXES[a];
            match ts.tool {
                Tool::Move | Tool::Scale => {
                    let (Some(p0), Some(p1)) = (project_px(cam, vp, origin), project_px(cam, vp, origin + dir * len)) else { continue };
                    painter.line_segment([pt(p0), pt(p1)], stroke);
                    if ts.tool == Tool::Move {
                        let d2 = (p1 - p0).normalize_or_zero();
                        let n = Vec2::new(-d2.y, d2.x);
                        let base = p1 - d2 * 14.0 * ppp;
                        painter.add(egui::Shape::convex_polygon(
                            vec![pt(p1 + d2 * 6.0 * ppp), pt(base + n * 6.0 * ppp), pt(base - n * 6.0 * ppp)],
                            col,
                            Stroke::NONE,
                        ));
                    } else {
                        painter.rect_filled(egui::Rect::from_center_size(pt(p1), egui::vec2(10.0, 10.0)), 0.0, col);
                    }
                }
                Tool::Rotate => {
                    let (u, v) = ring_basis(a, orient);
                    let mut prev: Option<Vec2> = None;
                    for k in 0..=64 {
                        let th = k as f32 / 64.0 * TAU;
                        let s = project_px(cam, vp, origin + len * (th.cos() * u + th.sin() * v));
                        if let (Some(pp), Some(s)) = (prev, s) {
                            painter.line_segment([pt(pp), pt(s)], stroke);
                        }
                        prev = s;
                    }
                }
                _ => {}
            }
        }
    }
}
