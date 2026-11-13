use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositorNodeKind {
    RenderLayers,
    BrightnessContrast,
    Exposure,
    Blur,
    Glare,
    Invert,
    Composite,
    Viewer,
}

impl CompositorNodeKind {
    pub fn title(self) -> &'static str {
        match self {
            Self::RenderLayers => "Render Layers",
            Self::BrightnessContrast => "Brightness / Contrast",
            Self::Exposure => "Exposure",
            Self::Blur => "Blur",
            Self::Glare => "Glare",
            Self::Invert => "Invert",
            Self::Composite => "Composite",
            Self::Viewer => "Viewer",
        }
    }

    pub fn is_output(self) -> bool {
        matches!(self, Self::Composite | Self::Viewer)
    }

    pub fn has_image_input(self) -> bool {
        !matches!(self, Self::RenderLayers)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompositorNode {
    pub id: u64,
    pub kind: CompositorNodeKind,
    pub pos: [f32; 2],
    pub value: f32,
    pub secondary: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositorLink {
    pub from_node: u64,
    pub to_node: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompositorGraph {
    pub nodes: Vec<CompositorNode>,
    pub links: Vec<CompositorLink>,
    pub next_id: u64,
}

impl Default for CompositorGraph {
    fn default() -> Self {
        Self {
            nodes: vec![
                CompositorNode {
                    id: 1,
                    kind: CompositorNodeKind::RenderLayers,
                    pos: [40.0, 100.0],
                    value: 0.0,
                    secondary: 0.0,
                },
                CompositorNode {
                    id: 2,
                    kind: CompositorNodeKind::Composite,
                    pos: [440.0, 80.0],
                    value: 0.0,
                    secondary: 0.0,
                },
                CompositorNode {
                    id: 3,
                    kind: CompositorNodeKind::Viewer,
                    pos: [440.0, 280.0],
                    value: 0.0,
                    secondary: 0.0,
                },
            ],
            links: vec![
                CompositorLink {
                    from_node: 1,
                    to_node: 2,
                },
                CompositorLink {
                    from_node: 1,
                    to_node: 3,
                },
            ],
            next_id: 4,
        }
    }
}

impl CompositorGraph {
    pub fn add_node(&mut self, kind: CompositorNodeKind, pos: [f32; 2]) -> u64 {
        let id = self.next_id.max(1);
        self.next_id = id.saturating_add(1);
        self.nodes.push(CompositorNode {
            id,
            kind,
            pos,
            value: 0.0,
            secondary: 0.0,
        });
        id
    }

    pub fn connect(&mut self, from_node: u64, to_node: u64) -> bool {
        if from_node == to_node
            || !self
                .nodes
                .iter()
                .any(|node| node.id == from_node && !node.kind.is_output())
            || !self
                .nodes
                .iter()
                .any(|node| node.id == to_node && node.kind.has_image_input())
            || self.links.iter().any(|link| link.to_node == to_node)
        {
            return false;
        }
        self.links.push(CompositorLink { from_node, to_node });
        if self.has_cycle() {
            self.links.pop();
            return false;
        }
        true
    }

    pub fn remove_node(&mut self, id: u64) {
        self.nodes.retain(|node| node.id != id);
        self.links
            .retain(|link| link.from_node != id && link.to_node != id);
    }

    pub fn evaluate(&self, width: u32, height: u32, render_rgba: &[u8]) -> Option<Vec<u8>> {
        let pixel_count = (width as usize).checked_mul(height as usize)?;
        if width == 0 || height == 0 || render_rgba.len() != pixel_count.checked_mul(4)? {
            return None;
        }
        let output_id = [CompositorNodeKind::Composite, CompositorNodeKind::Viewer]
            .into_iter()
            .find_map(|kind| {
                let id = self.nodes.iter().find(|node| node.kind == kind)?.id;
                self.links
                    .iter()
                    .any(|link| link.to_node == id)
                    .then_some(id)
            });
        let Some(output_id) = output_id else {
            return self
                .nodes
                .iter()
                .any(|node| node.kind == CompositorNodeKind::RenderLayers)
                .then(|| render_rgba.to_vec());
        };
        let Some(first_link) = self.links.iter().find(|link| link.to_node == output_id) else {
            return self
                .nodes
                .iter()
                .any(|node| node.kind == CompositorNodeKind::RenderLayers)
                .then(|| render_rgba.to_vec());
        };
        let mut chain = Vec::new();
        let mut current_id = first_link.from_node;
        let mut visited = std::collections::HashSet::new();
        loop {
            if !visited.insert(current_id) {
                return None;
            }
            let node = self.nodes.iter().find(|node| node.id == current_id)?;
            if node.kind == CompositorNodeKind::RenderLayers {
                break;
            }
            chain.push(node);
            current_id = self
                .links
                .iter()
                .find(|link| link.to_node == current_id)?
                .from_node;
        }
        let mut image = render_rgba.to_vec();
        for node in chain.into_iter().rev() {
            image = match node.kind {
                CompositorNodeKind::BrightnessContrast => {
                    brightness_contrast(&image, node.value, node.secondary)
                }
                CompositorNodeKind::Exposure => exposure(&image, node.value),
                CompositorNodeKind::Blur => box_blur(&image, width, height, node.value),
                CompositorNodeKind::Glare => {
                    glare(&image, width, height, node.value, node.secondary)
                }
                CompositorNodeKind::Invert => invert(&image),
                CompositorNodeKind::RenderLayers
                | CompositorNodeKind::Composite
                | CompositorNodeKind::Viewer => return None,
            };
        }
        Some(image)
    }

    fn has_cycle(&self) -> bool {
        fn visit(
            id: u64,
            graph: &CompositorGraph,
            visiting: &mut std::collections::HashSet<u64>,
            done: &mut std::collections::HashSet<u64>,
        ) -> bool {
            if done.contains(&id) {
                return false;
            }
            if !visiting.insert(id) {
                return true;
            }
            for link in graph.links.iter().filter(|link| link.from_node == id) {
                if visit(link.to_node, graph, visiting, done) {
                    return true;
                }
            }
            visiting.remove(&id);
            done.insert(id);
            false
        }

        let mut visiting = std::collections::HashSet::new();
        let mut done = std::collections::HashSet::new();
        self.nodes
            .iter()
            .any(|node| visit(node.id, self, &mut visiting, &mut done))
    }
}

fn brightness_contrast(src: &[u8], brightness: f32, contrast: f32) -> Vec<u8> {
    let c = contrast.clamp(-100.0, 100.0) * 2.55;
    let factor = (259.0 * (c + 255.0)) / (255.0 * (259.0 - c));
    src.chunks_exact(4)
        .flat_map(|pixel| {
            [
                (factor * (pixel[0] as f32 - 128.0) + 128.0 + brightness).clamp(0.0, 255.0) as u8,
                (factor * (pixel[1] as f32 - 128.0) + 128.0 + brightness).clamp(0.0, 255.0) as u8,
                (factor * (pixel[2] as f32 - 128.0) + 128.0 + brightness).clamp(0.0, 255.0) as u8,
                pixel[3],
            ]
        })
        .collect()
}

fn exposure(src: &[u8], stops: f32) -> Vec<u8> {
    let gain = 2.0_f32.powf(stops.clamp(-16.0, 16.0));
    src.chunks_exact(4)
        .flat_map(|pixel| {
            [
                (pixel[0] as f32 * gain).clamp(0.0, 255.0) as u8,
                (pixel[1] as f32 * gain).clamp(0.0, 255.0) as u8,
                (pixel[2] as f32 * gain).clamp(0.0, 255.0) as u8,
                pixel[3],
            ]
        })
        .collect()
}

fn invert(src: &[u8]) -> Vec<u8> {
    src.chunks_exact(4)
        .flat_map(|pixel| [255 - pixel[0], 255 - pixel[1], 255 - pixel[2], pixel[3]])
        .collect()
}

fn box_blur(src: &[u8], width: u32, height: u32, radius: f32) -> Vec<u8> {
    let radius = radius.round().clamp(0.0, 32.0) as i32;
    if radius == 0 {
        return src.to_vec();
    }
    let (width, height) = (width as usize, height as usize);
    let mut tmp = vec![0u8; src.len()];
    let mut out = vec![0u8; src.len()];
    for y in 0..height {
        let mut sum = [0u32; 4];
        let initial_end = radius.min(width as i32 - 1) as usize;
        for x in 0..=initial_end {
            add_pixel(src, &mut sum, y * width + x, 1);
        }
        for x in 0..width {
            let left = x.saturating_sub(radius as usize);
            let right = (x + radius as usize).min(width - 1);
            if x > 0 {
                let remove = x as i32 - radius - 1;
                let add = x as i32 + radius;
                if remove >= 0 {
                    add_pixel(src, &mut sum, y * width + remove as usize, -1);
                }
                if add < width as i32 {
                    add_pixel(src, &mut sum, y * width + add as usize, 1);
                }
            }
            write_average(&mut tmp, y * width + x, sum, right - left + 1);
        }
    }
    for x in 0..width {
        let mut sum = [0u32; 4];
        let initial_end = radius.min(height as i32 - 1) as usize;
        for y in 0..=initial_end {
            add_pixel(&tmp, &mut sum, y * width + x, 1);
        }
        for y in 0..height {
            let top = y.saturating_sub(radius as usize);
            let bottom = (y + radius as usize).min(height - 1);
            if y > 0 {
                let remove = y as i32 - radius - 1;
                let add = y as i32 + radius;
                if remove >= 0 {
                    add_pixel(&tmp, &mut sum, remove as usize * width + x, -1);
                }
                if add < height as i32 {
                    add_pixel(&tmp, &mut sum, add as usize * width + x, 1);
                }
            }
            write_average(&mut out, y * width + x, sum, bottom - top + 1);
        }
    }
    out
}

fn add_pixel(src: &[u8], sum: &mut [u32; 4], pixel: usize, direction: i32) {
    let offset = pixel * 4;
    for (channel, component) in sum.iter_mut().enumerate() {
        *component = (*component as i64 + src[offset + channel] as i64 * direction as i64) as u32;
    }
}

fn write_average(dst: &mut [u8], pixel: usize, sum: [u32; 4], count: usize) {
    let offset = pixel * 4;
    for (channel, value) in sum.into_iter().enumerate() {
        dst[offset + channel] = (value / count as u32) as u8;
    }
}

fn glare(src: &[u8], width: u32, height: u32, threshold: f32, strength: f32) -> Vec<u8> {
    let threshold = threshold.clamp(0.0, 1.0) * 255.0;
    let mut highlights = src.to_vec();
    for pixel in highlights.chunks_exact_mut(4) {
        let luminance =
            pixel[0] as f32 * 0.2126 + pixel[1] as f32 * 0.7152 + pixel[2] as f32 * 0.0722;
        if luminance < threshold {
            pixel[..3].fill(0);
        }
    }
    let bloom = box_blur(&highlights, width, height, 8.0);
    let gain = strength.clamp(0.0, 4.0);
    src.chunks_exact(4)
        .zip(bloom.chunks_exact(4))
        .flat_map(|(base, glow)| {
            [
                (base[0] as f32 + glow[0] as f32 * gain).clamp(0.0, 255.0) as u8,
                (base[1] as f32 + glow[1] as f32 * gain).clamp(0.0, 255.0) as u8,
                (base[2] as f32 + glow[2] as f32 * gain).clamp(0.0, 255.0) as u8,
                base[3],
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_graph_passes_render_image_through_composite() {
        let graph = CompositorGraph::default();
        let source = vec![20, 40, 80, 255];
        assert_eq!(graph.evaluate(1, 1, &source), Some(source));
    }

    #[test]
    fn graph_applies_effect_and_rejects_cycles() {
        let mut graph = CompositorGraph::default();
        let effect = graph.add_node(CompositorNodeKind::Invert, [220.0, 100.0]);
        graph.links.retain(|link| link.to_node != 2);
        assert!(graph.connect(1, effect));
        assert!(graph.connect(effect, 2));
        assert_eq!(
            graph.evaluate(1, 1, &[10, 20, 30, 255]),
            Some(vec![245, 235, 225, 255])
        );
        let second = graph.add_node(CompositorNodeKind::Exposure, [300.0, 100.0]);
        assert!(graph.connect(effect, second));
        graph.links.retain(|link| link.to_node != effect);
        assert!(!graph.connect(second, effect));
    }

    #[test]
    fn invalid_render_buffer_is_not_evaluated() {
        assert_eq!(CompositorGraph::default().evaluate(2, 1, &[0; 4]), None);
    }

    #[test]
    fn blur_uses_a_bounded_separable_box_filter() {
        assert_eq!(
            box_blur(&[0, 0, 0, 255, 30, 30, 30, 255, 60, 60, 60, 255], 3, 1, 1.0),
            vec![15, 15, 15, 255, 30, 30, 30, 255, 45, 45, 45, 255]
        );
    }

    #[test]
    fn neutral_brightness_and_contrast_is_identity() {
        let src = vec![12, 100, 240, 255];
        assert_eq!(brightness_contrast(&src, 0.0, 0.0), src);
    }
}
