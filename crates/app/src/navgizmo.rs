//! Blender-style viewport navigation gizmo: a clickable axis cross (snap the view to
//! an axis, drag to orbit) and, in Object Mode, buttons for zoom, pan, camera view and
//! perspective / orthographic.
use egui::{pos2, vec2, Align2, Color32, FontId, Pos2, Sense, Stroke};
use forge_core::OrbitCamera;
use glam::Vec3;
use std::f32::consts::{FRAC_PI_2, PI};

/// Pitch used for straight top / bottom views (just short of the Y-up singularity).
pub const TOP_PITCH: f32 = 1.5706;

pub enum NavAction {
    /// Look along an axis: target yaw / pitch of the orbit camera.
    SnapView { yaw: f32, pitch: f32 },
    /// Orbit by a mouse-style delta in physical pixels.
    Orbit(f32, f32),
    /// Pan by a mouse-style delta in physical pixels.
    Pan(f32, f32),
    /// Zoom by the same units as one mouse-wheel notch.
    Zoom(f32),
    ToggleCameraView,
    TogglePerspective,
}

const AXES: [(Vec3, Color32, &str); 3] = [
    (Vec3::X, Color32::from_rgb(237, 70, 95), "X"),
    (Vec3::Y, Color32::from_rgb(139, 220, 0), "Y"),
    (Vec3::Z, Color32::from_rgb(44, 143, 255), "Z"),
];

/// Yaw / pitch that views the scene from the given side of an axis.
fn snap_target(axis: usize, positive: bool) -> (f32, f32) {
    let sign = if positive { 1.0_f32 } else { -1.0_f32 };
    match axis {
        0 => (sign * FRAC_PI_2, 0.0),
        1 => (0.0, sign * TOP_PITCH),
        _ => (if positive { 0.0 } else { PI }, 0.0),
    }
}

fn angle_diff(a: f32, b: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let mut d = (a - b) % tau;
    if d > PI {
        d -= tau;
    } else if d < -PI {
        d += tau;
    }
    d
}

/// True when the camera already looks at the scene from that side of the axis.
fn is_at(cam: &OrbitCamera, axis: usize, positive: bool) -> bool {
    let (yaw, pitch) = snap_target(axis, positive);
    if axis == 1 {
        return (cam.pitch - pitch).abs() < 0.01;
    }
    angle_diff(cam.yaw, yaw).abs() < 0.01 && (cam.pitch - pitch).abs() < 0.01
}

struct Ball {
    axis: usize,
    positive: bool,
    pos: Pos2,
    depth: f32,
}

fn icon_zoom(p: &egui::Painter, c: Pos2, col: Color32) {
    let st = Stroke::new(1.7_f32, col);
    p.circle_stroke(c + vec2(-1.5, -1.5), 5.0, st);
    p.line_segment([c + vec2(2.2, 2.2), c + vec2(7.0, 7.0)], Stroke::new(2.2_f32, col));
}

fn icon_hand(p: &egui::Painter, c: Pos2, col: Color32) {
    let palm = egui::Rect::from_center_size(c + vec2(0.0, 3.0), vec2(11.0, 9.0));
    p.rect_filled(palm, 3.0_f32, col);
    for dx in [-4.0_f32, -1.4, 1.4, 4.0] {
        p.line_segment(
            [c + vec2(dx, 1.0), c + vec2(dx, -6.0)],
            Stroke::new(2.2_f32, col),
        );
    }
    p.line_segment(
        [c + vec2(-5.0, 4.0), c + vec2(-8.5, 0.5)],
        Stroke::new(2.2_f32, col),
    );
}

fn icon_camera(p: &egui::Painter, c: Pos2, col: Color32) {
    let st = Stroke::new(1.6_f32, col);
    let body = egui::Rect::from_center_size(c + vec2(0.0, 1.0), vec2(16.0, 10.0));
    p.rect_stroke(body, 2.0_f32, st);
    p.circle_stroke(c + vec2(0.0, 1.0), 3.0, st);
    p.line_segment([c + vec2(-4.0, -4.0), c + vec2(-1.0, -4.0)], Stroke::new(2.2_f32, col));
}

fn icon_grid(p: &egui::Painter, c: Pos2, col: Color32, ortho: bool) {
    let st = Stroke::new(1.3_f32, col);
    if ortho {
        // flat square grid
        for i in 0..4 {
            let o = -7.0 + i as f32 * 14.0 / 3.0;
            p.line_segment([c + vec2(o, -7.0), c + vec2(o, 7.0)], st);
            p.line_segment([c + vec2(-7.0, o), c + vec2(7.0, o)], st);
        }
    } else {
        // grid in perspective: converging lines
        for (top, bot) in [(-4.0_f32, -9.0_f32), (-1.3, -3.0), (1.3, 3.0), (4.0, 9.0)] {
            p.line_segment([c + vec2(top, -6.0), c + vec2(bot, 6.0)], st);
        }
        for (y, half) in [(-6.0_f32, 4.0_f32), (-1.0, 6.5), (6.0, 9.0)] {
            p.line_segment([c + vec2(-half, y), c + vec2(half, y)], st);
        }
    }
}

/// Draw the gizmo in the top-right corner of `rect` (the 3D viewport, in points).
/// `full` adds the Object-Mode buttons below the axis cross.
pub fn draw(
    ctx: &egui::Context,
    cam: &OrbitCamera,
    rect: egui::Rect,
    full: bool,
    in_camera_view: bool,
) -> Vec<NavAction> {
    let mut out = Vec::new();
    let ppp = ctx.pixels_per_point();
    let size = 92.0_f32;
    let btn = 28.0_f32;
    let pad = 10.0_f32;
    let pos = pos2(rect.right() - size - pad, rect.top() + pad);

    egui::Area::new(egui::Id::new("nav_gizmo"))
        .fixed_pos(pos)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            // ---------------------------------------------------------- axis cross
            let (resp, painter) =
                ui.allocate_painter(vec2(size, size), Sense::click_and_drag());
            let center = resp.rect.center();
            let ring = size * 0.5 - 15.0;
            let view = cam.view();

            let mut balls: Vec<Ball> = Vec::with_capacity(6);
            for (i, (axis, _, _)) in AXES.iter().enumerate() {
                for positive in [true, false] {
                    let a = if positive { *axis } else { -*axis };
                    let d = view.transform_vector3(a);
                    balls.push(Ball {
                        axis: i,
                        positive,
                        pos: center + vec2(d.x, -d.y) * ring,
                        depth: d.z,
                    });
                }
            }
            balls.sort_by(|a, b| a.depth.total_cmp(&b.depth));

            let pointer = resp.hover_pos();
            let radius_of = |b: &Ball| if b.positive { 9.5_f32 } else { 6.5_f32 };
            // Nearest-to-viewer ball under the pointer (balls are sorted far -> near).
            let hovered = pointer.and_then(|p| {
                balls
                    .iter()
                    .rposition(|b| (b.pos - p).length() <= radius_of(b) + 1.5)
            });

            if resp.hovered() || resp.dragged() {
                painter.circle_filled(
                    center,
                    size * 0.5 - 2.0,
                    Color32::from_white_alpha(28),
                );
            }

            for (idx, b) in balls.iter().enumerate() {
                let (_, col, label) = AXES[b.axis];
                let hot = hovered == Some(idx);
                let r = radius_of(b);
                if b.positive {
                    painter.line_segment(
                        [center, b.pos],
                        Stroke::new(2.0_f32, col.gamma_multiply(if hot { 1.0 } else { 0.85 })),
                    );
                    painter.circle_filled(b.pos, r, col);
                    if hot {
                        painter.circle_stroke(b.pos, r + 1.0, Stroke::new(1.5_f32, Color32::WHITE));
                    }
                    painter.text(
                        b.pos,
                        Align2::CENTER_CENTER,
                        label,
                        FontId::proportional(12.0),
                        Color32::from_black_alpha(220),
                    );
                } else {
                    let fill = if hot {
                        col.gamma_multiply(0.9)
                    } else {
                        col.gamma_multiply(0.38)
                    };
                    painter.circle_filled(b.pos, r, fill);
                    painter.circle_stroke(b.pos, r, Stroke::new(1.2_f32, col.gamma_multiply(0.8)));
                    if hot {
                        painter.text(
                            b.pos,
                            Align2::CENTER_CENTER,
                            format!("-{label}"),
                            FontId::proportional(9.0),
                            Color32::from_black_alpha(230),
                        );
                    }
                }
            }

            if resp.dragged() {
                let d = resp.drag_delta();
                out.push(NavAction::Orbit(d.x * ppp, d.y * ppp));
            } else if resp.clicked() {
                if let Some(idx) = hovered {
                    let b = &balls[idx];
                    // Clicking the side you already look from flips to the opposite side.
                    let positive = if b.positive && is_at(cam, b.axis, true) {
                        false
                    } else {
                        b.positive
                    };
                    let (yaw, pitch) = snap_target(b.axis, positive);
                    out.push(NavAction::SnapView { yaw, pitch });
                }
            }
            resp.on_hover_text("Click an axis to snap the view, drag to orbit");

            // ------------------------------------------------- Object Mode buttons
            if !full {
                return;
            }
            ui.add_space(6.0);
            let idle = Color32::from_rgb(210, 210, 215);
            let bg = Color32::from_black_alpha(150);
            for kind in 0..4 {
                let (row, _) = ui.allocate_exact_size(vec2(size, btn), Sense::hover());
                let rc = egui::Rect::from_center_size(row.center(), vec2(btn, btn));
                let id = ui.id().with(("nav_btn", kind));
                let sense = if kind < 2 {
                    Sense::click_and_drag()
                } else {
                    Sense::click()
                };
                let r = ui.interact(rc, id, sense);
                let active = kind == 2 && in_camera_view;
                let fill = if active {
                    Color32::from_rgb(52, 102, 180)
                } else if r.hovered() || r.dragged() {
                    Color32::from_rgba_unmultiplied(80, 80, 90, 210)
                } else {
                    bg
                };
                let p = ui.painter();
                p.circle_filled(rc.center(), btn * 0.5, fill);
                match kind {
                    0 => icon_zoom(p, rc.center(), idle),
                    1 => icon_hand(p, rc.center(), idle),
                    2 => icon_camera(p, rc.center(), idle),
                    _ => icon_grid(p, rc.center(), idle, cam.ortho),
                }
                match kind {
                    0 => {
                        if r.dragged() {
                            out.push(NavAction::Zoom(-r.drag_delta().y * 0.1));
                        }
                        r.on_hover_text("Zoom (drag up / down)");
                    }
                    1 => {
                        if r.dragged() {
                            let d = r.drag_delta();
                            out.push(NavAction::Pan(d.x * ppp, d.y * ppp));
                        }
                        r.on_hover_text("Pan (drag)");
                    }
                    2 => {
                        if r.clicked() {
                            out.push(NavAction::ToggleCameraView);
                        }
                        r.on_hover_text("Toggle camera view");
                    }
                    _ => {
                        if r.clicked() {
                            out.push(NavAction::TogglePerspective);
                        }
                        r.on_hover_text(if cam.ortho {
                            "Orthographic - click for perspective"
                        } else {
                            "Perspective - click for orthographic"
                        });
                    }
                }
                ui.add_space(4.0);
            }
        });
    out
}
