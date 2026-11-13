//! Equirectangular HDR environment maps + spherical-harmonics irradiance.
use glam::Vec3;
use std::f32::consts::{PI, TAU};

#[derive(Clone, Debug)]
pub struct EnvironmentMap {
    pub width: u32,
    pub height: u32,
    /// Linear RGB, 3 floats per pixel, row-major, top row = straight up (+Y).
    pub rgb: Vec<f32>,
}

/// Real SH basis (bands 0-2) for a unit direction.
pub fn sh_basis(d: Vec3) -> [f32; 9] {
    let (x, y, z) = (d.x, d.y, d.z);
    [
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl EnvironmentMap {
    pub fn uniform(width: u32, height: u32, c: [f32; 3]) -> Self {
        let mut rgb = Vec::with_capacity((width * height * 3) as usize);
        for _ in 0..width * height {
            rgb.extend_from_slice(&c);
        }
        Self { width, height, rgb }
    }

    /// Direction for texel (x, y). Must match `equirect_uv` in the shader.
    pub fn texel_dir(x: u32, y: u32, w: u32, h: u32) -> Vec3 {
        let phi = ((x as f32 + 0.5) / w as f32 - 0.5) * TAU;
        let theta = (y as f32 + 0.5) / h as f32 * PI;
        Vec3::new(
            theta.sin() * phi.cos(),
            theta.cos(),
            theta.sin() * phi.sin(),
        )
    }

    /// Clear-sky gradient + sun + ground bounce. Used when no HDRI is loaded.
    pub fn procedural_sky(width: u32, height: u32) -> Self {
        let sun = Vec3::new(0.45, 0.75, 0.35).normalize();
        let horizon = Vec3::new(0.80, 0.85, 0.92);
        let zenith = Vec3::new(0.18, 0.38, 0.80);
        let ground = Vec3::new(0.16, 0.15, 0.14);
        let mut rgb = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                let d = Self::texel_dir(x, y, width, height);
                let mut c = if d.y >= 0.0 {
                    horizon.lerp(zenith, d.y.powf(0.5))
                } else {
                    (horizon * 0.45).lerp(ground, (-d.y).powf(0.4))
                };
                if d.y > -0.05 {
                    let cs = d.dot(sun);
                    c += Vec3::splat(
                        6.0 * ((cs - 1.0) * 250.0).exp() + 30.0 * smoothstep(0.9985, 0.9998, cs),
                    );
                }
                rgb.extend_from_slice(&c.to_array());
            }
        }
        Self { width, height, rgb }
    }

    /// 2x2 box downsample (dimensions halve, minimum 1).
    pub fn downsample(&self) -> Self {
        let (w, h) = (self.width, self.height);
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut rgb = Vec::with_capacity((nw * nh * 3) as usize);
        for y in 0..nh {
            for x in 0..nw {
                let xs = [(x * 2).min(w - 1), (x * 2 + 1).min(w - 1)];
                let ys = [(y * 2).min(h - 1), (y * 2 + 1).min(h - 1)];
                for c in 0..3 {
                    let mut s = 0.0;
                    for &sy in &ys {
                        for &sx in &xs {
                            s += self.rgb[((sy * w + sx) * 3 + c) as usize];
                        }
                    }
                    rgb.push(s * 0.25);
                }
            }
        }
        Self {
            width: nw,
            height: nh,
            rgb,
        }
    }

    /// Level 0 (self) down to 1x1.
    pub fn mip_chain(&self) -> Vec<EnvironmentMap> {
        let mut v = vec![self.clone()];
        while {
            let l = v.last().unwrap();
            l.width > 1 || l.height > 1
        } {
            let next = v.last().unwrap().downsample();
            v.push(next);
        }
        v
    }

    /// 9 SH coefficients (per RGB) of the *cosine-convolved* radiance, already divided by pi:
    /// `diffuse_light(n) = sum_i c_i * sh_basis(n)[i]`, then multiply by albedo.
    pub fn sh9(&self) -> [[f32; 3]; 9] {
        let mut acc = [[0.0f64; 3]; 9];
        let (w, h) = (self.width, self.height);
        for y in 0..h {
            let theta = (y as f32 + 0.5) / h as f32 * PI;
            let d_omega = (TAU / w as f32) * (PI / h as f32) * theta.sin();
            for x in 0..w {
                let basis = sh_basis(Self::texel_dir(x, y, w, h));
                let i = ((y * w + x) * 3) as usize;
                for k in 0..9 {
                    for c in 0..3 {
                        acc[k][c] += (self.rgb[i + c] * basis[k] * d_omega) as f64;
                    }
                }
            }
        }
        // A_l / pi : band0 = 1, band1 = 2/3, band2 = 1/4
        let band = [
            1.0,
            2.0 / 3.0,
            2.0 / 3.0,
            2.0 / 3.0,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
        ];
        let mut out = [[0.0f32; 3]; 9];
        for k in 0..9 {
            for c in 0..3 {
                out[k][c] = acc[k][c] as f32 * band[k];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_white_env_gives_unit_diffuse_light() {
        let env = EnvironmentMap::uniform(64, 32, [1.0; 3]);
        let sh = env.sh9();
        for n in [Vec3::Y, Vec3::X, Vec3::new(1.0, 1.0, 1.0).normalize()] {
            let v: f32 = sh_basis(n)
                .iter()
                .zip(sh.iter())
                .map(|(y, c)| y * c[0])
                .sum();
            assert!((v - 1.0).abs() < 0.02, "got {v}");
        }
    }

    #[test]
    fn mip_chain_ends_at_1x1() {
        let c = EnvironmentMap::uniform(64, 32, [1.0; 3]).mip_chain();
        assert_eq!(c.len(), 7);
        assert_eq!((c.last().unwrap().width, c.last().unwrap().height), (1, 1));
    }
}
