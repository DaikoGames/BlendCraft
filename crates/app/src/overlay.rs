//! Viewport overlays: helper icons (empties, cameras, lights), the 3D cursor,
//! selection shapes (box / lasso / circle) and the modal-transform status text.
use crate::tools::{project_px, SelShape};
use egui::{Color32, Stroke};
use forge_core::{helper_lines, Document, EditDomain, OrbitCamera};
use glam::{Mat4, Vec2, Vec3, Vec4};

pub struct Extra<'a> {
    pub shape: Option<&'a SelShape>,
    pub modal: Option<&'a str>,
    /// circle brush: center and radius in physical pixels
    pub circle: Option<(Vec2, f32)>,
    /// Measure tool endpoints in world space
    pub measure: Option<(Vec3, Vec3)>,
    /// Annotation strokes in world space
    pub annotations: &'a [Vec<Vec3>],
    pub annotate_stroke: Option<&'a [Vec3]>,
}

/// Project a world segment to physical pixels, clipping against the near plane.
fn clip_project(m: Mat4, vp: [f32; 4], a: Vec3, b: Vec3) -> Option<(Vec2, Vec2)> {
    let mut c0 = m * a.extend(1.0);
    let mut c1 = m * b.extend(1.0);
    let eps = 1e-3;
    if c0.w < eps && c1.w < eps {
        return None;
    }
    if c0.w < eps {
        let t = (eps - c0.w) / (c1.w - c0.w);
        c0 += (c1 - c0) * t;
    } else if c1.w < eps {
        let t = (eps - c1.w) / (c0.w - c1.w);
        c1 += (c0 - c1) * t;
    }
    let s = |c: Vec4| Vec2::new(vp[0] + (c.x / c.w * 0.5 + 0.5) * vp[2], vp[1] + (1.0 - (c.y / c.w * 0.5 + 0.5)) * vp[3]);
    Some((s(c0), s(c1)))
}

pub fn draw(ctx: &egui::Context, cam: &OrbitCamera, rect: egui::Rect, doc: &Document, extra: &Extra) {
    let ppp = ctx.pixels_per_point();
    let vp = [rect.min.x * ppp, rect.min.y * ppp, rect.width() * ppp, rect.height() * ppp];
    let pt = |p: Vec2| egui::pos2(p.x / ppp, p.y / ppp);
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Middle, egui::Id::new("helpers"))).with_clip_rect(rect);

    // helper icons for non-mesh objects
    if doc.scene.objects.iter().any(|o| !o.kind.is_mesh()) {
        let m = cam.view_proj(vp[2] / vp[3]);
        let worlds = doc.scene.world_matrices();
        for (i, o) in doc.scene.objects.iter().enumerate() {
            if !doc.scene.is_visible(i) || o.kind.is_mesh() {
                continue;
            }
            let sel = doc.is_selected(i);
            let active = doc.active == Some(o.id) && sel;
            let col = if active {
                Color32::from_rgb(255, 190, 60)
            } else if sel {
                Color32::from_rgb(235, 100, 20)
            } else {
                Color32::from_rgb(190, 190, 205)
            };
            let stroke = Stroke::new(if sel { 2.0_f32 } else { 1.4_f32 }, col);
            for (a, b) in helper_lines(o, &worlds[i]) {
                if let Some((p0, p1)) = clip_project(m, vp, a, b) {
                    painter.line_segment([pt(p0), pt(p1)], stroke);
                }
            }
        }
    }

    // 3D cursor
    if let Some(c) = project_px(cam, vp, doc.scene.cursor) {
        let c = pt(c);
        painter.circle_stroke(c, 10.0, Stroke::new(3.0_f32, Color32::WHITE));
        painter.circle_stroke(c, 10.0, Stroke::new(1.5_f32, Color32::from_rgb(230, 40, 40)));
        for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
            painter.line_segment([c + egui::vec2(dx * 6.0, dy * 6.0), c + egui::vec2(dx * 15.0, dy * 15.0)], Stroke::new(1.5_f32, Color32::WHITE));
        }
    }

    // selection shapes
    let sel_stroke = Stroke::new(1.5_f32, Color32::from_rgb(255, 255, 255));
    match extra.shape {
        Some(SelShape::Box { a, b }) => {
            let r = egui::Rect::from_two_pos(pt(*a), pt(*b));
            painter.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(255, 255, 255, 25));
            painter.rect_stroke(r, 0.0, sel_stroke);
        }
        Some(SelShape::Lasso(pts)) if pts.len() > 1 => {
            let mut line: Vec<egui::Pos2> = pts.iter().map(|p| pt(*p)).collect();
            line.push(line[0]);
            painter.add(egui::Shape::line(line, sel_stroke));
        }
        _ => {}
    }
    if let Some((c, r)) = extra.circle {
        painter.circle_stroke(pt(c), r / ppp, sel_stroke);
    }

    // Measure line
    if let Some((a, b)) = extra.measure {
        let m = cam.view_proj(vp[2] / vp[3].max(1.0));
        if let Some((pa, pb)) = clip_project(m, vp, a, b) {
            let stroke = Stroke::new(2.0_f32, Color32::from_rgb(255, 220, 50));
            painter.line_segment([pt(pa), pt(pb)], stroke);
            painter.circle_filled(pt(pa), 4.0, Color32::from_rgb(255, 220, 50));
            painter.circle_filled(pt(pb), 4.0, Color32::from_rgb(255, 220, 50));
            let mid = (pa + pb) * 0.5;
            let dist = (b - a).length();
            painter.text(
                pt(mid) + egui::vec2(0.0, -12.0),
                egui::Align2::CENTER_BOTTOM,
                format!("{:.3} m", dist),
                egui::FontId::proportional(13.0),
                Color32::from_rgb(255, 240, 120),
            );
        }
    }

    // Annotations
    let ann_stroke = Stroke::new(2.0_f32, Color32::from_rgb(80, 200, 255));
    let draw_stroke = |stroke: &[Vec3]| {
        let mut pts = Vec::new();
        for w in stroke.iter() {
            if let Some(p) = project_px(cam, vp, *w) {
                pts.push(pt(p));
            }
        }
        if pts.len() >= 2 {
            painter.add(egui::Shape::line(pts, ann_stroke));
        }
    };
    for stroke in extra.annotations {
        draw_stroke(stroke);
    }
    if let Some(stroke) = extra.annotate_stroke {
        draw_stroke(stroke);
    }


    // Edit Mode: vertices / edges / faces
    if let Some(session) = doc.edit.as_ref() {
        if let Some(idx) = doc.scene.objects.iter().position(|o| o.id == session.object_id) {
            let world = doc.scene.world_matrices()[idx];
            let verts: Vec<Vec3> = session
                .mesh
                .vertices
                .iter()
                .map(|v| world.transform_point3(Vec3::from(v.position)))
                .collect();
            let selected_verts = session.selected_vert_indices();
            // All mesh edges (topology) — shows points are connected
            let edge_col = Color32::from_rgb(90, 160, 220);
            let edge_sel = Color32::from_rgb(255, 180, 40);
            for (a, b) in session.build_edges() {
                let (Some(p0), Some(p1)) = (
                    project_px(cam, vp, verts[a as usize]),
                    project_px(cam, vp, verts[b as usize]),
                ) else { continue };
                let edge_selected = session.selection.edges.contains(&(a, b))
                    || session.selection.edges.contains(&(b, a));
                let both_sel = selected_verts.contains(&a) && selected_verts.contains(&b);
                let sel = edge_selected || both_sel;
                let stroke = Stroke::new(
                    if sel { 2.5_f32 } else { 1.5_f32 },
                    if sel { edge_sel } else { edge_col },
                );
                painter.line_segment([pt(p0), pt(p1)], stroke);
            }
            // Vertices as connected joints
            for (i, w) in verts.iter().enumerate() {
                let Some(p) = project_px(cam, vp, *w) else { continue };
                let sel = selected_verts.contains(&(i as u32));
                let r = if sel { 6.5 } else { 4.5 };
                let fill = if sel {
                    Color32::from_rgb(255, 210, 50)
                } else {
                    Color32::from_rgb(30, 30, 35)
                };
                painter.circle_filled(pt(p), r / ppp, fill);
                painter.circle_stroke(
                    pt(p),
                    r / ppp,
                    Stroke::new(1.2_f32, if sel { Color32::WHITE } else { Color32::from_rgb(180, 200, 230) }),
                );
            }
            // face centers for face mode
            if session.selection.domain == EditDomain::Face {
                for (fi, tri) in session.mesh.indices.chunks_exact(3).enumerate() {
                    let c = (verts[tri[0] as usize] + verts[tri[1] as usize] + verts[tri[2] as usize]) / 3.0;
                    let Some(p) = project_px(cam, vp, c) else { continue };
                    let sel = session.selection.faces.contains(&(fi as u32));
                    let col = if sel {
                        Color32::from_rgba_unmultiplied(255, 160, 0, 200)
                    } else {
                        Color32::from_rgba_unmultiplied(80, 80, 200, 120)
                    };
                    painter.circle_filled(pt(p), 4.0 / ppp, col);
                }
            }
            // mode badge
            let label = match session.selection.domain {
                EditDomain::Vertex => "Edit Mode — Vertex".to_string(),
                EditDomain::Edge => "Edit Mode — Edge (lines)".to_string(),
                EditDomain::Face => "Edit Mode — Face".to_string(),
            };
            let galley = ctx.fonts(|f| {
                f.layout_no_wrap(label, egui::FontId::proportional(14.0), Color32::from_rgb(255, 200, 80))
            });
            let pos = egui::pos2(rect.min.x + 12.0, rect.min.y + 12.0);
            painter.rect_filled(
                egui::Rect::from_min_size(pos, galley.size()).expand(6.0),
                4.0,
                Color32::from_black_alpha(180),
            );
            painter.galley(pos, galley, Color32::from_rgb(255, 200, 80));
        }
    }

    // modal status line
    if let Some(text) = extra.modal {
        let galley = ctx.fonts(|f| f.layout_no_wrap(text.to_string(), egui::FontId::proportional(15.0), Color32::WHITE));
        let size = galley.size();
        let pos = egui::pos2(rect.center().x - size.x * 0.5, rect.max.y - size.y - 18.0);
        painter.rect_filled(egui::Rect::from_min_size(pos, size).expand(7.0), 5.0, Color32::from_black_alpha(190));
        painter.galley(pos, galley, Color32::WHITE);
    }
}
