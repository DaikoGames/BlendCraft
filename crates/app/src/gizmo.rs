//! Transform gizmo: Move (axis arrows), Rotate (axis rings), Scale (axis handles).
//! Works on the whole selection, honours pivot point, orientation and snapping.
use crate::tools::{project_px, SnapMode, Tool, ToolState};
use crate::transform::{closest_param, cursor_ray, orientation, pivot_point, vertex_snap, Op, Session, AXES};
use egui::{Color32, Stroke};
use forge_core::{Document, OrbitCamera};
use glam::{Quat, Vec2, Vec3};
use std::collections::HashSet;
use std::f32::consts::TAU;

struct Drag {
    axis: usize,
    axis_dir: Vec3,
    orient: Quat,
    origin: Vec3,
    /// Move/Scale: start parameter along the axis. Rotate: start angle.
    start: f32,
    tool: Tool,
    session: Session,
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
    fn frame(&self, ts: &ToolState, cam: &OrbitCamera, doc: &Document) -> Option<(Vec3, Quat, f32)> {
        if !ts.tool.is_gizmo() {
            return None;
        }
        let mode = if ts.pivot == crate::tools::Pivot::Individual { crate::tools::Pivot::Active } else { ts.pivot };
        let origin = pivot_point(doc, mode)?;
        let orient = orientation(doc, cam, ts.orient);
        let len = (cam.eye() - origin).length().max(0.01) * 0.15;
        Some((origin, orient, len))
    }

    fn hit_axis(&self, tool: Tool, cursor: (f64, f64), cam: &OrbitCamera, vp: [f32; 4], origin: Vec3, orient: Quat, len: f32) -> Option<usize> {
        let c = Vec2::new(cursor.0 as f32, cursor.1 as f32);
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
        let axis_dir = orient * AXES[axis];
        let (ro, rd) = cursor_ray(cursor, cam, vp);
        let start = match ts.tool {
            Tool::Rotate => plane_angle(ro, rd, origin, axis_dir, orient, axis),
            _ => closest_param(ro, rd, origin, axis_dir),
        };
        let Some(start) = start else { return false };
        let Some(session) = Session::new(doc, ts.pivot) else { return false };
        let exclude = session.exclude.clone();
        doc.checkpoint();
        self.drag = Some(Drag { axis, axis_dir, orient, origin, start, tool: ts.tool, session, exclude });
        true
    }

    pub fn update(&mut self, cursor: (f64, f64), cam: &OrbitCamera, vp: [f32; 4], doc: &mut Document, ts: &ToolState, ctrl: bool) {
        let Some(d) = self.drag.as_ref() else { return };
        let snapping = ts.snap.enabled ^ ctrl;
        let (ro, rd) = cursor_ray(cursor, cam, vp);
        let op = match d.tool {
            Tool::Move => {
                let Some(t) = closest_param(ro, rd, d.origin, d.axis_dir) else { return };
                let mut delta = t - d.start;
                if snapping {
                    match ts.snap.mode {
                        SnapMode::Increment => {
                            let st = ts.snap.move_step.max(1e-4);
                            delta = (delta / st).round() * st;
                        }
                        SnapMode::Vertex => {
                            if let Some(p) = vertex_snap(doc, cam, vp, cursor, &d.exclude) {
                                delta = (p - d.origin).dot(d.axis_dir);
                            }
                        }
                    }
                }
                Op::Move(d.axis_dir * delta)
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
                let Some(t) = closest_param(ro, rd, d.origin, d.axis_dir) else { return };
                if d.start.abs() < 1e-4 {
                    return;
                }
                let mut f = (t / d.start).max(0.01);
                if snapping {
                    let st = ts.snap.scale_step.max(1e-3);
                    f = ((f / st).round() * st).max(st);
                }
                Op::Scale { axis: Some(d.axis_dir), factor: f }
            }
            _ => return,
        };
        d.session.apply(doc, op);
    }

    /// Draws the handles. `rect` is the 3D viewport in egui points.
    pub fn draw(&self, ctx: &egui::Context, cam: &OrbitCamera, rect: egui::Rect, doc: &Document, ts: &ToolState) {
        let Some((origin, orient, len)) = self.frame(ts, cam, doc) else { return };
        let ppp = ctx.pixels_per_point();
        let vp = [rect.min.x * ppp, rect.min.y * ppp, rect.width() * ppp, rect.height() * ppp];
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("gizmo"))).with_clip_rect(rect);
        let pt = |p: Vec2| egui::pos2(p.x / ppp, p.y / ppp);
        let active = self.drag.as_ref().map(|d| d.axis).or(self.hover);

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
