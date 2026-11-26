//! Node compositor modeled on Blender's Compositor (manual overview):
//! Render Layers → filter/color nodes → Composite (final) / Viewer (preview).
//! Evaluation is a DAG over image sockets, not a single forced chain.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompositorNodeKind {
    /// Scene render result (Blender: Render Layers — Image output).
    RenderLayers,
    /// Final file/display composite (Blender: Composite).
    Composite,
    /// Preview (Blender: Viewer).
    Viewer,
    BrightnessContrast,
    Exposure,
    Gamma,
    HueSaturation,
    Invert,
    RGB,
    Mix,
    AlphaOver,
    Blur,
    Glare,
    /// Blender Filter > Sobel-ish edge enhance
    Filter,
    Flip,
    SetAlpha,
    ColorBalance,
    Tonemap,
    Posterize,
}

impl CompositorNodeKind {
    pub fn title(self) -> &'static str {
        match self {
            Self::RenderLayers => "Render Layers",
            Self::Composite => "Composite",
            Self::Viewer => "Viewer",
            Self::BrightnessContrast => "Brightness/Contrast",
            Self::Exposure => "Exposure",
            Self::Gamma => "Gamma",
            Self::HueSaturation => "Hue/Saturation",
            Self::Invert => "Invert",
            Self::RGB => "RGB",
            Self::Mix => "Mix",
            Self::AlphaOver => "Alpha Over",
            Self::Blur => "Blur",
            Self::Glare => "Glare",
            Self::Filter => "Filter (Edges)",
            Self::Flip => "Flip",
            Self::SetAlpha => "Set Alpha",
            Self::ColorBalance => "Color Balance",
            Self::Tonemap => "Tonemap",
            Self::Posterize => "Posterize",
        }
    }

    pub fn is_output(self) -> bool {
        matches!(self, Self::Composite | Self::Viewer)
    }

    /// Number of image inputs (Blender-style Image / Image sockets).
    pub fn image_inputs(self) -> u8 {
        match self {
            Self::RenderLayers | Self::RGB => 0,
            Self::Mix | Self::AlphaOver | Self::SetAlpha => 2,
            _ => 1,
        }
    }

    pub fn has_image_output(self) -> bool {
        !self.is_output()
    }

    pub fn input_socket_name(self, index: u8) -> &'static str {
        match self {
            Self::Mix => match index {
                0 => "Image",
                1 => "Image",
                _ => "Image",
            },
            Self::AlphaOver => match index {
                0 => "Image",
                1 => "Image",
                _ => "Image",
            },
            Self::SetAlpha => match index {
                0 => "Image",
                1 => "Alpha",
                _ => "Image",
            },
            Self::Composite | Self::Viewer => "Image",
            _ => "Image",
        }
    }

    pub fn output_socket_name(self) -> &'static str {
        match self {
            Self::RenderLayers => "Image",
            Self::RGB => "Image",
            _ if self.is_output() => "",
            _ => "Image",
        }
    }

    pub fn category(self) -> &'static str {
        match self {
            Self::RenderLayers | Self::RGB => "Input",
            Self::Composite | Self::Viewer => "Output",
            Self::Mix | Self::AlphaOver | Self::SetAlpha => "Color Mix",
            Self::Blur | Self::Glare | Self::Filter => "Filter",
            Self::Flip => "Distort",
            _ => "Color",
        }
    }

    /// Labels for value knobs shown in the UI (Blender-ish defaults).
    pub fn value_labels(self) -> (&'static str, Option<&'static str>) {
        match self {
            Self::BrightnessContrast => ("Bright", Some("Contrast")),
            Self::Exposure => ("Exposure", None),
            Self::Gamma => ("Gamma", None),
            Self::HueSaturation => ("Hue", Some("Saturation")),
            Self::RGB => ("R/G (use 2nd for B)", Some("Blue")),
            Self::Mix => ("Fac", None),
            Self::AlphaOver => ("Fac", None),
            Self::Blur => ("Size", None),
            Self::Glare => ("Threshold", Some("Strength")),
            Self::Filter => ("Strength", None),
            Self::Flip => ("Flip X", Some("Flip Y")),
            Self::SetAlpha => ("Fac", None),
            Self::ColorBalance => ("Lift", Some("Gain")),
            Self::Tonemap => ("Key", Some("Offset")),
            Self::Posterize => ("Steps", None),
            _ => ("Value", None),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompositorNode {
    pub id: u64,
    pub kind: CompositorNodeKind,
    pub pos: [f32; 2],
    /// Primary scalar (Fac, Exposure, Size, …).
    pub value: f32,
    /// Secondary scalar (Contrast, Saturation, Strength, …).
    pub secondary: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositorLink {
    pub from_node: u64,
    #[serde(default)]
    pub from_socket: u8,
    pub to_node: u64,
    #[serde(default)]
    pub to_socket: u8,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompositorGraph {
    pub nodes: Vec<CompositorNode>,
    pub links: Vec<CompositorLink>,
    pub next_id: u64,
}

impl Default for CompositorGraph {
    fn default() -> Self {
        // Blender default-ish: Render Layers → Composite + Viewer
        Self {
            nodes: vec![
                CompositorNode {
                    id: 1,
                    kind: CompositorNodeKind::RenderLayers,
                    pos: [40.0, 120.0],
                    value: 0.0,
                    secondary: 0.0,
                },
                CompositorNode {
                    id: 2,
                    kind: CompositorNodeKind::Composite,
                    pos: [480.0, 60.0],
                    value: 0.0,
                    secondary: 0.0,
                },
                CompositorNode {
                    id: 3,
                    kind: CompositorNodeKind::Viewer,
                    pos: [480.0, 260.0],
                    value: 0.0,
                    secondary: 0.0,
                },
            ],
            links: vec![
                CompositorLink {
                    from_node: 1,
                    from_socket: 0,
                    to_node: 2,
                    to_socket: 0,
                },
                CompositorLink {
                    from_node: 1,
                    from_socket: 0,
                    to_node: 3,
                    to_socket: 0,
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
        let (value, secondary) = default_params(kind);
        self.nodes.push(CompositorNode {
            id,
            kind,
            pos,
            value,
            secondary,
        });
        id
    }

    /// Connect image sockets. Replaces any existing link into the same to_socket.
    pub fn connect_sockets(
        &mut self,
        from_node: u64,
        from_socket: u8,
        to_node: u64,
        to_socket: u8,
    ) -> bool {
        if from_node == to_node {
            return false;
        }
        let Some(from) = self.nodes.iter().find(|n| n.id == from_node) else {
            return false;
        };
        let Some(to) = self.nodes.iter().find(|n| n.id == to_node) else {
            return false;
        };
        if !from.kind.has_image_output() || to_socket >= to.kind.image_inputs() {
            return false;
        }
        self.links
            .retain(|l| !(l.to_node == to_node && l.to_socket == to_socket));
        self.links.push(CompositorLink {
            from_node,
            from_socket,
            to_node,
            to_socket,
        });
        if self.has_cycle() {
            self.links.pop();
            return false;
        }
        true
    }

    /// Back-compat: single image socket connect (socket 0).
    pub fn connect(&mut self, from_node: u64, to_node: u64) -> bool {
        self.connect_sockets(from_node, 0, to_node, 0)
    }

    pub fn remove_node(&mut self, id: u64) {
        self.nodes.retain(|n| n.id != id);
        self.links
            .retain(|l| l.from_node != id && l.to_node != id);
    }

    /// Prefer Viewer for backdrop/preview; fall back to Composite (Blender behaviour).
    pub fn evaluate(&self, width: u32, height: u32, render_rgba: &[u8]) -> Option<Vec<u8>> {
        self.evaluate_target(width, height, render_rgba, PreferOutput::ViewerThenComposite)
    }

    pub fn evaluate_composite(
        &self,
        width: u32,
        height: u32,
        render_rgba: &[u8],
    ) -> Option<Vec<u8>> {
        self.evaluate_target(width, height, render_rgba, PreferOutput::CompositeOnly)
    }

    fn evaluate_target(
        &self,
        width: u32,
        height: u32,
        render_rgba: &[u8],
        prefer: PreferOutput,
    ) -> Option<Vec<u8>> {
        let pixel_count = (width as usize).checked_mul(height as usize)?;
        if width == 0 || height == 0 || render_rgba.len() != pixel_count.checked_mul(4)? {
            return None;
        }

        let order = self.topo_order()?;
        let mut images: HashMap<u64, Vec<u8>> = HashMap::new();

        for id in order {
            let node = self.nodes.iter().find(|n| n.id == id)?;
            let out = match node.kind {
                CompositorNodeKind::RenderLayers => render_rgba.to_vec(),
                CompositorNodeKind::RGB => solid_rgb(width, height, node.value, node.secondary),
                CompositorNodeKind::Composite | CompositorNodeKind::Viewer => {
                    // Pass-through of Image input (socket 0).
                    self.input_image(id, 0, &images, render_rgba, width, height)
                }
                CompositorNodeKind::BrightnessContrast => {
                    let src = self.input_image(id, 0, &images, render_rgba, width, height);
                    brightness_contrast(&src, node.value, node.secondary)
                }
                CompositorNodeKind::Exposure => {
                    let src = self.input_image(id, 0, &images, render_rgba, width, height);
                    exposure(&src, node.value)
                }
                CompositorNodeKind::Gamma => {
                    let src = self.input_image(id, 0, &images, render_rgba, width, height);
                    gamma_node(&src, node.value)
                }
                CompositorNodeKind::HueSaturation => {
                    let src = self.input_image(id, 0, &images, render_rgba, width, height);
                    hue_sat(&src, node.value, node.secondary)
                }
                CompositorNodeKind::Invert => {
                    let src = self.input_image(id, 0, &images, render_rgba, width, height);
                    invert(&src)
                }
                CompositorNodeKind::Blur => {
                    let src = self.input_image(id, 0, &images, render_rgba, width, height);
                    box_blur(&src, width, height, node.value)
                }
                CompositorNodeKind::Glare => {
                    let src = self.input_image(id, 0, &images, render_rgba, width, height);
                    glare(&src, width, height, node.value, node.secondary)
                }
                CompositorNodeKind::Filter => {
                    let src = self.input_image(id, 0, &images, render_rgba, width, height);
                    edge_filter(&src, width, height, node.value)
                }
                CompositorNodeKind::Flip => {
                    let src = self.input_image(id, 0, &images, render_rgba, width, height);
                    flip_image(&src, width, height, node.value > 0.5, node.secondary > 0.5)
                }
                CompositorNodeKind::SetAlpha => {
                    let img = self.input_image(id, 0, &images, render_rgba, width, height);
                    let alpha = self.input_image(id, 1, &images, render_rgba, width, height);
                    set_alpha(&img, &alpha, node.value)
                }
                CompositorNodeKind::ColorBalance => {
                    let src = self.input_image(id, 0, &images, render_rgba, width, height);
                    color_balance(&src, node.value, node.secondary)
                }
                CompositorNodeKind::Tonemap => {
                    let src = self.input_image(id, 0, &images, render_rgba, width, height);
                    tonemap(&src, node.value, node.secondary)
                }
                CompositorNodeKind::Posterize => {
                    let src = self.input_image(id, 0, &images, render_rgba, width, height);
                    posterize(&src, node.value)
                }
                CompositorNodeKind::Mix => {
                    let a = self.input_image(id, 0, &images, render_rgba, width, height);
                    let b = self.input_image(id, 1, &images, render_rgba, width, height);
                    mix_images(&a, &b, node.value)
                }
                CompositorNodeKind::AlphaOver => {
                    let bg = self.input_image(id, 0, &images, render_rgba, width, height);
                    let fg = self.input_image(id, 1, &images, render_rgba, width, height);
                    alpha_over(&bg, &fg, node.value)
                }
            };
            images.insert(id, out);
        }

        let pick = |kind: CompositorNodeKind| {
            self.nodes
                .iter()
                .find(|n| n.kind == kind)
                .and_then(|n| images.get(&n.id).cloned())
        };

        match prefer {
            PreferOutput::ViewerThenComposite => pick(CompositorNodeKind::Viewer)
                .or_else(|| pick(CompositorNodeKind::Composite))
                .or_else(|| Some(render_rgba.to_vec())),
            PreferOutput::CompositeOnly => pick(CompositorNodeKind::Composite)
                .or_else(|| pick(CompositorNodeKind::Viewer))
                .or_else(|| Some(render_rgba.to_vec())),
        }
    }

    fn input_image(
        &self,
        to_node: u64,
        to_socket: u8,
        images: &HashMap<u64, Vec<u8>>,
        render_rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        if let Some(link) = self
            .links
            .iter()
            .find(|l| l.to_node == to_node && l.to_socket == to_socket)
        {
            if let Some(img) = images.get(&link.from_node) {
                return img.clone();
            }
        }
        // Unconnected Image input → black transparent (Blender-ish), except socket 0
        // on outputs where we fall back to the render for convenience.
        if to_socket == 0 {
            render_rgba.to_vec()
        } else {
            vec![0u8; (width as usize) * (height as usize) * 4]
        }
    }

    fn topo_order(&self) -> Option<Vec<u64>> {
        let mut indeg: HashMap<u64, usize> = HashMap::new();
        for n in &self.nodes {
            indeg.insert(n.id, 0);
        }
        for l in &self.links {
            *indeg.entry(l.to_node).or_insert(0) += 1;
        }
        let mut q: VecDeque<u64> = indeg
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| *id)
            .collect();
        let mut out = Vec::with_capacity(self.nodes.len());
        while let Some(id) = q.pop_front() {
            out.push(id);
            for l in self.links.iter().filter(|l| l.from_node == id) {
                if let Some(d) = indeg.get_mut(&l.to_node) {
                    *d = d.saturating_sub(1);
                    if *d == 0 {
                        q.push_back(l.to_node);
                    }
                }
            }
        }
        if out.len() == self.nodes.len() {
            Some(out)
        } else {
            None
        }
    }

    fn has_cycle(&self) -> bool {
        self.topo_order().is_none()
    }
}

enum PreferOutput {
    ViewerThenComposite,
    CompositeOnly,
}

fn default_params(kind: CompositorNodeKind) -> (f32, f32) {
    match kind {
        CompositorNodeKind::BrightnessContrast => (0.0, 0.0),
        CompositorNodeKind::Exposure => (0.0, 0.0),
        CompositorNodeKind::Gamma => (1.0, 0.0),
        CompositorNodeKind::HueSaturation => (0.5, 1.0),
        CompositorNodeKind::RGB => (0.5, 0.5), // R stored as value? we pack R=value G=secondary later differently
        CompositorNodeKind::Mix | CompositorNodeKind::AlphaOver => (1.0, 0.0),
        CompositorNodeKind::Blur => (4.0, 0.0),
        CompositorNodeKind::Glare => (0.8, 1.0),
        CompositorNodeKind::Filter => (1.0, 0.0),
        CompositorNodeKind::Flip => (1.0, 0.0),
        CompositorNodeKind::SetAlpha => (1.0, 0.0),
        CompositorNodeKind::ColorBalance => (0.0, 1.0),
        CompositorNodeKind::Tonemap => (0.18, 1.0),
        CompositorNodeKind::Posterize => (8.0, 0.0),
        _ => (0.0, 0.0),
    }
}

fn solid_rgb(width: u32, height: u32, rg: f32, b: f32) -> Vec<u8> {
    // value = average of R/G for compact UI; secondary = B
    let r = (rg.clamp(0.0, 1.0) * 255.0) as u8;
    let g = r;
    let bb = (b.clamp(0.0, 1.0) * 255.0) as u8;
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..(width * height) {
        out.extend_from_slice(&[r, g, bb, 255]);
    }
    out
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

fn gamma_node(src: &[u8], gamma: f32) -> Vec<u8> {
    let g = gamma.clamp(0.01, 10.0);
    let inv = 1.0 / g;
    src.chunks_exact(4)
        .flat_map(|pixel| {
            [
                ((pixel[0] as f32 / 255.0).powf(inv) * 255.0).clamp(0.0, 255.0) as u8,
                ((pixel[1] as f32 / 255.0).powf(inv) * 255.0).clamp(0.0, 255.0) as u8,
                ((pixel[2] as f32 / 255.0).powf(inv) * 255.0).clamp(0.0, 255.0) as u8,
                pixel[3],
            ]
        })
        .collect()
}

fn hue_sat(src: &[u8], hue: f32, sat: f32) -> Vec<u8> {
    // Blender HSV: hue offset around 0.5 = neutral, sat 1.0 = neutral.
    let hue_shift = (hue.clamp(0.0, 1.0) - 0.5) * std::f32::consts::TAU;
    let sat_mul = sat.clamp(0.0, 2.0);
    src.chunks_exact(4)
        .flat_map(|pixel| {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            let (h, s, v) = rgb_to_hsv(r, g, b);
            let h2 = (h + hue_shift / std::f32::consts::TAU).rem_euclid(1.0);
            let s2 = (s * sat_mul).clamp(0.0, 1.0);
            let (nr, ng, nb) = hsv_to_rgb(h2, s2, v);
            [
                (nr * 255.0).clamp(0.0, 255.0) as u8,
                (ng * 255.0).clamp(0.0, 255.0) as u8,
                (nb * 255.0).clamp(0.0, 255.0) as u8,
                pixel[3],
            ]
        })
        .collect()
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let v = max;
    let d = max - min;
    let s = if max <= 1e-6 { 0.0 } else { d / max };
    let h = if d <= 1e-6 {
        0.0
    } else if max == r {
        ((g - b) / d).rem_euclid(6.0) / 6.0
    } else if max == g {
        (((b - r) / d) + 2.0) / 6.0
    } else {
        (((r - g) / d) + 4.0) / 6.0
    };
    (h, s, v)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

fn invert(src: &[u8]) -> Vec<u8> {
    src.chunks_exact(4)
        .flat_map(|pixel| [255 - pixel[0], 255 - pixel[1], 255 - pixel[2], pixel[3]])
        .collect()
}

fn mix_images(a: &[u8], b: &[u8], fac: f32) -> Vec<u8> {
    let f = fac.clamp(0.0, 1.0);
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .flat_map(|(pa, pb)| {
            [
                (pa[0] as f32 * (1.0 - f) + pb[0] as f32 * f).clamp(0.0, 255.0) as u8,
                (pa[1] as f32 * (1.0 - f) + pb[1] as f32 * f).clamp(0.0, 255.0) as u8,
                (pa[2] as f32 * (1.0 - f) + pb[2] as f32 * f).clamp(0.0, 255.0) as u8,
                (pa[3] as f32 * (1.0 - f) + pb[3] as f32 * f).clamp(0.0, 255.0) as u8,
            ]
        })
        .collect()
}

fn alpha_over(bg: &[u8], fg: &[u8], fac: f32) -> Vec<u8> {
    // Blender Alpha Over: fg over bg, Fac scales fg alpha.
    let f = fac.clamp(0.0, 1.0);
    bg.chunks_exact(4)
        .zip(fg.chunks_exact(4))
        .flat_map(|(b, g)| {
            let a = (g[3] as f32 / 255.0) * f;
            let ia = 1.0 - a;
            [
                (g[0] as f32 * a + b[0] as f32 * ia).clamp(0.0, 255.0) as u8,
                (g[1] as f32 * a + b[1] as f32 * ia).clamp(0.0, 255.0) as u8,
                (g[2] as f32 * a + b[2] as f32 * ia).clamp(0.0, 255.0) as u8,
                ((g[3] as f32 * f + b[3] as f32 * ia).clamp(0.0, 255.0)) as u8,
            ]
        })
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
    let count = count.max(1) as u32;
    for (channel, value) in sum.into_iter().enumerate() {
        dst[offset + channel] = (value / count) as u8;
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


fn edge_filter(src: &[u8], width: u32, height: u32, strength: f32) -> Vec<u8> {
    let s = strength.clamp(0.0, 4.0);
    let (w, h) = (width as usize, height as usize);
    let mut out = src.to_vec();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let idx = |xx: usize, yy: usize| (yy * w + xx) * 4;
            for c in 0..3 {
                let gx = src[idx(x + 1, y) + c] as i32 - src[idx(x - 1, y) + c] as i32;
                let gy = src[idx(x, y + 1) + c] as i32 - src[idx(x, y - 1) + c] as i32;
                let edge = ((gx.abs() + gy.abs()) as f32 * s).clamp(0.0, 255.0);
                let base = src[idx(x, y) + c] as f32;
                out[idx(x, y) + c] = (base * 0.5 + edge * 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

fn flip_image(src: &[u8], width: u32, height: u32, flip_x: bool, flip_y: bool) -> Vec<u8> {
    if !flip_x && !flip_y {
        return src.to_vec();
    }
    let (w, h) = (width as usize, height as usize);
    let mut out = vec![0u8; src.len()];
    for y in 0..h {
        for x in 0..w {
            let sx = if flip_x { w - 1 - x } else { x };
            let sy = if flip_y { h - 1 - y } else { y };
            let si = (sy * w + sx) * 4;
            let di = (y * w + x) * 4;
            out[di..di + 4].copy_from_slice(&src[si..si + 4]);
        }
    }
    out
}

fn set_alpha(img: &[u8], alpha_src: &[u8], fac: f32) -> Vec<u8> {
    let f = fac.clamp(0.0, 1.0);
    img.chunks_exact(4)
        .zip(alpha_src.chunks_exact(4))
        .flat_map(|(p, a)| {
            let lum = a[0] as f32 * 0.2126 + a[1] as f32 * 0.7152 + a[2] as f32 * 0.0722;
            let alpha = (lum * f).clamp(0.0, 255.0) as u8;
            [p[0], p[1], p[2], alpha]
        })
        .collect()
}

fn color_balance(src: &[u8], lift: f32, gain: f32) -> Vec<u8> {
    // Simplified lift/gain (Blender Color Balance subset).
    let lift = lift.clamp(-1.0, 1.0);
    let gain = gain.clamp(0.0, 4.0);
    src.chunks_exact(4)
        .flat_map(|p| {
            let mut rgb = [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0];
            for c in &mut rgb {
                *c = ((*c + lift) * gain).clamp(0.0, 1.0);
            }
            [
                (rgb[0] * 255.0) as u8,
                (rgb[1] * 255.0) as u8,
                (rgb[2] * 255.0) as u8,
                p[3],
            ]
        })
        .collect()
}

fn tonemap(src: &[u8], key: f32, offset: f32) -> Vec<u8> {
    // Simple Reinhard-style tonemap on LDR buffer.
    let key = key.clamp(0.01, 2.0);
    let offset = offset.clamp(0.01, 4.0);
    src.chunks_exact(4)
        .flat_map(|p| {
            let mut rgb = [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0];
            for c in &mut rgb {
                let x = *c * key;
                *c = (x / (1.0 + x) * offset).clamp(0.0, 1.0);
            }
            [
                (rgb[0] * 255.0) as u8,
                (rgb[1] * 255.0) as u8,
                (rgb[2] * 255.0) as u8,
                p[3],
            ]
        })
        .collect()
}

fn posterize(src: &[u8], steps: f32) -> Vec<u8> {
    let steps = steps.round().clamp(2.0, 32.0);
    src.chunks_exact(4)
        .flat_map(|p| {
            let quant = |v: u8| {
                let t = v as f32 / 255.0;
                ((t * (steps - 1.0)).round() / (steps - 1.0) * 255.0) as u8
            };
            [quant(p[0]), quant(p[1]), quant(p[2]), p[3]]
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
        // Rewire both Composite (2) and Viewer (3); evaluate() prefers Viewer.
        graph.links.retain(|link| link.to_node != 2 && link.to_node != 3);
        assert!(graph.connect(1, effect));
        assert!(graph.connect(effect, 2));
        assert!(graph.connect(effect, 3));
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
    fn mix_blends_two_inputs() {
        let mut graph = CompositorGraph::default();
        let rgb = graph.add_node(CompositorNodeKind::RGB, [100.0, 300.0]);
        let mix = graph.add_node(CompositorNodeKind::Mix, [300.0, 200.0]);
        graph.nodes.iter_mut().find(|n| n.id == mix).unwrap().value = 0.5;
        graph.links.clear();
        assert!(graph.connect_sockets(1, 0, mix, 0));
        assert!(graph.connect_sockets(rgb, 0, mix, 1));
        assert!(graph.connect_sockets(mix, 0, 2, 0));
        assert!(graph.connect_sockets(mix, 0, 3, 0));
        let out = graph.evaluate(1, 1, &[0, 0, 0, 255]).unwrap();
        assert!(out[0] > 0);
    }

    #[test]
    fn invalid_render_buffer_is_not_evaluated() {
        assert_eq!(CompositorGraph::default().evaluate(2, 1, &[0; 4]), None);
    }
}
