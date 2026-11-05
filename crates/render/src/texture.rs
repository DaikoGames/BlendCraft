//! CPU mip generation (sRGB-aware) and texture upload.

pub struct MipLevel {
    pub w: u32,
    pub h: u32,
    pub data: Vec<u8>,
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn downsample(src: &MipLevel, srgb: bool) -> MipLevel {
    let lut: Vec<f32> = (0..256).map(|i| srgb_to_linear(i as f32 / 255.0)).collect();
    let (nw, nh) = ((src.w / 2).max(1), (src.h / 2).max(1));
    let mut out = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..nh {
        let ys = [(y * 2).min(src.h - 1), (y * 2 + 1).min(src.h - 1)];
        for x in 0..nw {
            let xs = [(x * 2).min(src.w - 1), (x * 2 + 1).min(src.w - 1)];
            for c in 0..4usize {
                let mut sum = 0.0f32;
                for &sy in &ys {
                    for &sx in &xs {
                        let v = src.data[((sy * src.w + sx) as usize) * 4 + c];
                        sum += if srgb && c < 3 { lut[v as usize] } else { v as f32 / 255.0 };
                    }
                }
                let avg = sum * 0.25;
                let enc = if srgb && c < 3 { linear_to_srgb(avg) } else { avg };
                out[((y * nw + x) as usize) * 4 + c] = (enc * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
    MipLevel { w: nw, h: nh, data: out }
}

pub fn mip_chain_rgba8(w: u32, h: u32, data: Vec<u8>, srgb: bool) -> Vec<MipLevel> {
    let mut levels = vec![MipLevel { w, h, data }];
    while levels.last().map_or(false, |l| l.w > 1 || l.h > 1) {
        let next = downsample(levels.last().unwrap(), srgb);
        levels.push(next);
    }
    levels
}

/// Uploads the chain, skipping leading levels that exceed the device's texture size limit.
pub fn upload_mips(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    levels: &[MipLevel],
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    let max = device.limits().max_texture_dimension_2d;
    let start = levels.iter().position(|l| l.w <= max && l.h <= max).unwrap_or(levels.len() - 1);
    let levels = &levels[start..];
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: levels[0].w, height: levels[0].h, depth_or_array_layers: 1 },
        mip_level_count: levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (i, l) in levels.iter().enumerate() {
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: i as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &l.data,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4 * l.w), rows_per_image: Some(l.h) },
            wgpu::Extent3d { width: l.w, height: l.h, depth_or_array_layers: 1 },
        );
    }
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}
