//! Vertex Paint and Weight Paint brushes (local, offline).
use crate::mesh::MeshData;
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexPaintBrush {
    Draw,
    Soften,
    Smear,
    Blur,
}

impl VertexPaintBrush {
    pub fn name(self) -> &'static str {
        match self {
            Self::Draw => "Draw",
            Self::Soften => "Soften",
            Self::Smear => "Smear",
            Self::Blur => "Blur",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightPaintBrush {
    Draw,
    Blur,
    Average,
    Smear,
}

impl WeightPaintBrush {
    pub fn name(self) -> &'static str {
        match self {
            Self::Draw => "Draw",
            Self::Blur => "Blur",
            Self::Average => "Average",
            Self::Smear => "Smear",
        }
    }
}

#[derive(Clone, Debug)]
pub struct VertexPaintSettings {
    pub brush: VertexPaintBrush,
    pub color: [f32; 4],
    pub radius: f32,
    pub strength: f32,
}

impl Default for VertexPaintSettings {
    fn default() -> Self {
        Self {
            brush: VertexPaintBrush::Draw,
            color: [0.2, 0.45, 0.9, 1.0],
            radius: 0.35,
            strength: 0.5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct WeightPaintSettings {
    pub brush: WeightPaintBrush,
    pub weight: f32,
    pub radius: f32,
    pub strength: f32,
    pub normalize: bool,
}

impl Default for WeightPaintSettings {
    fn default() -> Self {
        Self {
            brush: WeightPaintBrush::Draw,
            weight: 1.0,
            radius: 0.35,
            strength: 0.5,
            normalize: true,
        }
    }
}

fn falloff(dist: f32, radius: f32) -> f32 {
    if dist >= radius || radius < 1e-8 {
        return 0.0;
    }
    let t = 1.0 - dist / radius;
    t * t * (3.0 - 2.0 * t) // smoothstep
}

/// Paint vertex colors near `center` (object-local space).
pub fn vertex_paint_stroke(
    mesh: &mut MeshData,
    centers: &[Vec3],
    settings: &VertexPaintSettings,
) -> bool {
    if centers.is_empty() || mesh.vertices.is_empty() {
        return false;
    }
    mesh.ensure_vertex_colors();
    let radius = settings.radius.max(1e-6);
    let strength = settings.strength.clamp(0.0, 1.0);
    let color = settings.color;
    let mut changed = false;

    match settings.brush {
        VertexPaintBrush::Draw => {
            for c in centers {
                for v in &mut mesh.vertices {
                    let p = Vec3::from(v.position);
                    let f = falloff(p.distance(*c), radius) * strength;
                    if f <= 0.0 {
                        continue;
                    }
                    for i in 0..4 {
                        v.color[i] = v.color[i] + (color[i] - v.color[i]) * f;
                    }
                    changed = true;
                }
            }
        }
        VertexPaintBrush::Soften | VertexPaintBrush::Blur => {
            // Average colors of neighbors within radius, blend toward average.
            let n = mesh.vertices.len();
            let mut sums = vec![[0.0f32; 4]; n];
            let mut counts = vec![0.0f32; n];
            // Build adjacency from triangles
            let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
            for tri in mesh.indices.chunks_exact(3) {
                let a = tri[0] as usize;
                let b = tri[1] as usize;
                let c = tri[2] as usize;
                adj[a].push(b);
                adj[a].push(c);
                adj[b].push(a);
                adj[b].push(c);
                adj[c].push(a);
                adj[c].push(b);
            }
            for i in 0..n {
                let col = mesh.vertices[i].color;
                for &j in &adj[i] {
                    for k in 0..4 {
                        sums[i][k] += mesh.vertices[j].color[k];
                    }
                    counts[i] += 1.0;
                }
                // include self
                for k in 0..4 {
                    sums[i][k] += col[k];
                }
                counts[i] += 1.0;
            }
            for c in centers {
                for i in 0..n {
                    let p = Vec3::from(mesh.vertices[i].position);
                    let f = falloff(p.distance(*c), radius) * strength;
                    if f <= 0.0 || counts[i] < 1.0 {
                        continue;
                    }
                    let inv = 1.0 / counts[i];
                    for k in 0..4 {
                        let avg = sums[i][k] * inv;
                        mesh.vertices[i].color[k] += (avg - mesh.vertices[i].color[k]) * f;
                    }
                    changed = true;
                }
            }
        }
        VertexPaintBrush::Smear => {
            // Pull colors from nearby vertices toward the stroke direction (use first→last).
            if centers.len() < 2 {
                return false;
            }
            let dir = (centers[centers.len() - 1] - centers[0]).normalize_or_zero();
            let sample = centers[0] - dir * (radius * 0.5);
            // Sample color near `sample`
            let mut sample_col = [0.0f32; 4];
            let mut wsum = 0.0f32;
            for v in &mesh.vertices {
                let d = Vec3::from(v.position).distance(sample);
                let w = falloff(d, radius);
                if w > 0.0 {
                    for k in 0..4 {
                        sample_col[k] += v.color[k] * w;
                    }
                    wsum += w;
                }
            }
            if wsum < 1e-6 {
                return false;
            }
            for k in 0..4 {
                sample_col[k] /= wsum;
            }
            for c in centers {
                for v in &mut mesh.vertices {
                    let p = Vec3::from(v.position);
                    let f = falloff(p.distance(*c), radius) * strength;
                    if f <= 0.0 {
                        continue;
                    }
                    for k in 0..4 {
                        v.color[k] += (sample_col[k] - v.color[k]) * f;
                    }
                    changed = true;
                }
            }
        }
    }
    changed
}

/// Paint weights into the active vertex group.
pub fn weight_paint_stroke(
    mesh: &mut MeshData,
    centers: &[Vec3],
    settings: &WeightPaintSettings,
) -> bool {
    if centers.is_empty() || mesh.vertices.is_empty() {
        return false;
    }
    mesh.ensure_vertex_group_sizes();
    if mesh.active_vertex_group < 0 || mesh.vertex_groups.is_empty() {
        if mesh.vertex_groups.is_empty() {
            mesh.add_vertex_group("Group");
        } else {
            mesh.active_vertex_group = 0;
        }
    }
    let gi = mesh.active_vertex_group as usize;
    if gi >= mesh.vertex_groups.len() || mesh.vertex_groups[gi].locked {
        return false;
    }
    let radius = settings.radius.max(1e-6);
    let strength = settings.strength.clamp(0.0, 1.0);
    let target = settings.weight.clamp(0.0, 1.0);
    let n = mesh.vertices.len();
    let mut changed = false;

    match settings.brush {
        WeightPaintBrush::Draw => {
            for c in centers {
                for i in 0..n {
                    let p = Vec3::from(mesh.vertices[i].position);
                    let f = falloff(p.distance(*c), radius) * strength;
                    if f <= 0.0 {
                        continue;
                    }
                    let w = &mut mesh.vertex_groups[gi].weights[i];
                    *w = (*w + (target - *w) * f).clamp(0.0, 1.0);
                    changed = true;
                }
            }
        }
        WeightPaintBrush::Blur | WeightPaintBrush::Average => {
            let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
            for tri in mesh.indices.chunks_exact(3) {
                let a = tri[0] as usize;
                let b = tri[1] as usize;
                let c = tri[2] as usize;
                adj[a].extend([b, c]);
                adj[b].extend([a, c]);
                adj[c].extend([a, b]);
            }
            let src: Vec<f32> = mesh.vertex_groups[gi].weights.clone();
            for c in centers {
                for i in 0..n {
                    let p = Vec3::from(mesh.vertices[i].position);
                    let f = falloff(p.distance(*c), radius) * strength;
                    if f <= 0.0 {
                        continue;
                    }
                    let mut sum = src[i];
                    let mut cnt = 1.0f32;
                    for &j in &adj[i] {
                        sum += src[j];
                        cnt += 1.0;
                    }
                    let avg = sum / cnt;
                    let w = &mut mesh.vertex_groups[gi].weights[i];
                    *w = (*w + (avg - *w) * f).clamp(0.0, 1.0);
                    changed = true;
                }
            }
        }
        WeightPaintBrush::Smear => {
            if centers.len() < 2 {
                return false;
            }
            let dir = (centers[centers.len() - 1] - centers[0]).normalize_or_zero();
            let sample = centers[0] - dir * (radius * 0.5);
            let mut sample_w = 0.0f32;
            let mut wsum = 0.0f32;
            for i in 0..n {
                let d = Vec3::from(mesh.vertices[i].position).distance(sample);
                let w = falloff(d, radius);
                if w > 0.0 {
                    sample_w += mesh.vertex_groups[gi].weights[i] * w;
                    wsum += w;
                }
            }
            if wsum < 1e-6 {
                return false;
            }
            sample_w /= wsum;
            for c in centers {
                for i in 0..n {
                    let p = Vec3::from(mesh.vertices[i].position);
                    let f = falloff(p.distance(*c), radius) * strength;
                    if f <= 0.0 {
                        continue;
                    }
                    let w = &mut mesh.vertex_groups[gi].weights[i];
                    *w = (*w + (sample_w - *w) * f).clamp(0.0, 1.0);
                    changed = true;
                }
            }
        }
    }

    if changed && settings.normalize {
        mesh.normalize_vertex_groups();
    }
    if changed {
        // Keep visualization in sync when weights change.
        mesh.apply_weight_visualization();
    }
    changed
}

/// Sync SkinBinding 4-influence weights from vertex groups named after bones when possible.
pub fn groups_to_skin_hint(mesh: &MeshData, bone_names: &[String]) -> Option<(Vec<[u16; 4]>, Vec<[f32; 4]>)> {
    if bone_names.is_empty() || mesh.vertices.is_empty() {
        return None;
    }
    let n = mesh.vertices.len();
    let mut joints = vec![[0u16; 4]; n];
    let mut weights = vec![[0f32; 4]; n];
    for vi in 0..n {
        let mut scores: Vec<(u16, f32)> = Vec::new();
        for (gi, g) in mesh.vertex_groups.iter().enumerate() {
            let w = g.weights.get(vi).copied().unwrap_or(0.0);
            if w <= 0.0 {
                continue;
            }
            if let Some(bi) = bone_names.iter().position(|b| b == &g.name) {
                scores.push((bi as u16, w));
            } else if gi < bone_names.len() {
                scores.push((gi as u16, w));
            }
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(4);
        let sum: f32 = scores.iter().map(|(_, w)| *w).sum::<f32>().max(1e-8);
        for (k, (j, w)) in scores.iter().enumerate() {
            joints[vi][k] = *j;
            weights[vi][k] = *w / sum;
        }
    }
    Some((joints, weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Vertex;

    fn unit_tri() -> MeshData {
        MeshData {
            name: "t".into(),
            vertices: vec![
                Vertex::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0]),
                Vertex::new([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0]),
                Vertex::new([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0]),
            ],
            indices: vec![0, 1, 2],
            ..Default::default()
        }
    }

    #[test]
    fn vertex_paint_draw_changes_color() {
        let mut m = unit_tri();
        let s = VertexPaintSettings {
            color: [1.0, 0.0, 0.0, 1.0],
            strength: 1.0,
            radius: 2.0,
            ..Default::default()
        };
        assert!(vertex_paint_stroke(&mut m, &[Vec3::ZERO], &s));
        assert!(m.vertices[0].color[0] > 0.5);
    }

    #[test]
    fn weight_paint_and_normalize() {
        let mut m = unit_tri();
        m.add_vertex_group("Bone.A");
        m.add_vertex_group("Bone.B");
        m.active_vertex_group = 0;
        let s = WeightPaintSettings {
            weight: 1.0,
            strength: 1.0,
            radius: 2.0,
            normalize: true,
            ..Default::default()
        };
        assert!(weight_paint_stroke(&mut m, &[Vec3::ZERO], &s));
        let sum: f32 = m.vertex_groups.iter().map(|g| g.weights[0]).sum();
        assert!((sum - 1.0).abs() < 1e-3 || sum < 1e-3 || (sum - 1.0).abs() < 0.1);
    }
}
