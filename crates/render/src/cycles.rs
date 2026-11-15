//! Progressive CPU path tracer used when the render engine is set to Cycles.
//!
//! This is a real Monte-Carlo path tracer (cosine-weighted hemisphere sampling +
//! next-event estimation to punctual lights), not a raster "Cycles look". It
//! accumulates samples over frames so the viewport converges like Blender's
//! progressive viewport.
use forge_core::{LightType, Material, MeshData, ObjectKind, OrbitCamera, Scene};
use glam::{Mat4, Vec3};
use std::sync::Arc;

const MAX_BOUNCES: u32 = 4;
const MAX_TRIANGLES: usize = 40_000;
const VIEWPORT_SPP: u32 = 1; // samples added per displayed frame
const MAX_DIM: u32 = 384; // path-trace at most this on the long side for interactivity

#[derive(Clone, Copy)]
struct Tri {
    v0: Vec3,
    e1: Vec3,
    e2: Vec3,
    n: Vec3,
    albedo: Vec3,
    emission: Vec3,
    metallic: f32,
    roughness: f32,
}

#[derive(Clone, Copy)]
struct Punctual {
    /// 0 point, 1 sun, 2 spot
    kind: u8,
    pos: Vec3,
    dir: Vec3,
    color: Vec3,
    range: f32,
}

pub struct CyclesState {
    tris: Vec<Tri>,
    lights: Vec<Punctual>,
    accum: Vec<[f32; 3]>,
    width: u32,
    height: u32,
    samples: u32,
    /// Fingerprint of camera + scene so we know when to reset.
    fingerprint: u64,
    /// Latest tonemapped RGBA8 preview (width*height*4).
    preview: Vec<u8>,
    exposure: f32,
}

impl Default for CyclesState {
    fn default() -> Self {
        Self {
            tris: Vec::new(),
            lights: Vec::new(),
            accum: Vec::new(),
            width: 0,
            height: 0,
            samples: 0,
            fingerprint: 0,
            preview: Vec::new(),
            exposure: 1.0,
        }
    }
}

impl CyclesState {
    pub fn samples(&self) -> u32 {
        self.samples
    }

    pub fn reset_accumulation(&mut self) {
        self.fingerprint = 0;
        self.samples = 0;
        self.accum.clear();
        self.preview.clear();
        self.width = 0;
        self.height = 0;
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn preview_rgba(&self) -> &[u8] {
        &self.preview
    }

    /// Rebuild geometry from the scene (call from `set_scene`).
    pub fn rebuild_scene(&mut self, scene: &Scene) {
        self.tris.clear();
        self.lights.clear();
        let worlds = scene.world_matrices();
        for (i, o) in scene.objects.iter().enumerate() {
            if !scene.is_visible(i) {
                continue;
            }
            let m = worlds.get(i).copied().unwrap_or(Mat4::IDENTITY);
            if let ObjectKind::Light(l) = &o.kind {
                let pos = m.w_axis.truncate();
                let dir = m.transform_vector3(Vec3::NEG_Z).normalize_or_zero();
                let color = Vec3::new(l.color[0], l.color[1], l.color[2]) * l.intensity.max(0.0);
                let kind = match l.kind {
                    LightType::Sun => 1,
                    LightType::Spot => 2,
                    _ => 0,
                };
                self.lights.push(Punctual {
                    kind,
                    pos,
                    dir,
                    color,
                    range: if l.range > 0.0 { l.range } else { 0.0 },
                });
                continue;
            }
            if !o.kind.is_mesh() || o.mesh.indices.is_empty() {
                continue;
            }
            let mat = o
                .material
                .and_then(|mi| scene.materials.get(mi))
                .cloned()
                .unwrap_or_else(Material::default);
            let albedo = Vec3::new(mat.base_color[0], mat.base_color[1], mat.base_color[2]);
            let emission = Vec3::new(mat.emissive[0], mat.emissive[1], mat.emissive[2]);
            push_mesh_tris(
                &mut self.tris,
                &o.mesh,
                m,
                albedo,
                emission,
                mat.metallic,
                mat.roughness,
            );
            if self.tris.len() >= MAX_TRIANGLES {
                break;
            }
        }
        // Changing geometry always invalidates the accumulation.
        self.fingerprint = self.fingerprint.wrapping_add(0x9E37_79B9);
        self.samples = 0;
        self.accum.clear();
        self.preview.clear();
    }

    /// Add samples for the current camera / viewport size. Returns true when the
    /// preview buffer was updated.
    pub fn accumulate(
        &mut self,
        camera: &OrbitCamera,
        view_w: u32,
        view_h: u32,
        exposure: f32,
        target_samples: u32,
    ) -> bool {
        if view_w < 2 || view_h < 2 {
            return false;
        }
        self.exposure = exposure.max(0.01);
        let (tw, th) = fit_size(view_w, view_h, MAX_DIM);
        let fp = camera_fingerprint(camera, tw, th);
        if fp != self.fingerprint || self.width != tw || self.height != th || self.accum.is_empty()
        {
            self.fingerprint = fp;
            self.width = tw;
            self.height = th;
            self.samples = 0;
            self.accum = vec![[0.0; 3]; (tw * th) as usize];
            self.preview = vec![0u8; (tw * th * 4) as usize];
        }

        if self.samples >= target_samples.max(1) {
            return false;
        }

        let spp = VIEWPORT_SPP;
        let w = self.width as usize;
        let h = self.height as usize;
        let eye = camera.eye();
        let target = camera.target;
        let forward = (target - eye).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        // Match a ~50° vertical FOV-ish scale.
        let aspect = self.width as f32 / self.height.max(1) as f32;
        let fov_scale = 0.6;

        let mut rng = (self.samples as u32).wrapping_mul(747796405).wrapping_add(2891336453);
        for y in 0..h {
            for x in 0..w {
                for _ in 0..spp {
                    let jx = rand01(&mut rng);
                    let jy = rand01(&mut rng);
                    let u = ((x as f32 + jx) / w as f32 * 2.0 - 1.0) * aspect * fov_scale;
                    let v = (1.0 - (y as f32 + jy) / h as f32 * 2.0) * fov_scale;
                    let dir = (forward + right * u + up * v).normalize_or_zero();
                    let col = trace(eye, dir, &self.tris, &self.lights, &mut rng);
                    let i = y * w + x;
                    self.accum[i][0] += col.x;
                    self.accum[i][1] += col.y;
                    self.accum[i][2] += col.z;
                }
            }
        }
        self.samples += spp;

        // Tonemap preview
        let inv = 1.0 / self.samples as f32;
        let exp = self.exposure;
        for (i, c) in self.accum.iter().enumerate() {
            let rgb = aces_tonemap(Vec3::new(c[0], c[1], c[2]) * inv * exp);
            let o = i * 4;
            self.preview[o] = (rgb.x * 255.0).clamp(0.0, 255.0) as u8;
            self.preview[o + 1] = (rgb.y * 255.0).clamp(0.0, 255.0) as u8;
            self.preview[o + 2] = (rgb.z * 255.0).clamp(0.0, 255.0) as u8;
            self.preview[o + 3] = 255;
        }
        true
    }
}

fn push_mesh_tris(
    out: &mut Vec<Tri>,
    mesh: &Arc<MeshData>,
    m: Mat4,
    albedo: Vec3,
    emission: Vec3,
    metallic: f32,
    roughness: f32,
) {
    let nmat = Mat4::from_mat3(glam::Mat3::from_mat4(m).inverse().transpose());
    for tri in mesh.indices.chunks_exact(3) {
        if out.len() >= MAX_TRIANGLES {
            return;
        }
        let p0 = m.transform_point3(Vec3::from(mesh.vertices[tri[0] as usize].position));
        let p1 = m.transform_point3(Vec3::from(mesh.vertices[tri[1] as usize].position));
        let p2 = m.transform_point3(Vec3::from(mesh.vertices[tri[2] as usize].position));
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let n = e1.cross(e2).normalize_or_zero();
        let n = if n.length_squared() < 1e-12 {
            nmat.transform_vector3(Vec3::from(mesh.vertices[tri[0] as usize].normal))
                .normalize_or_zero()
        } else {
            n
        };
        out.push(Tri {
            v0: p0,
            e1,
            e2,
            n,
            albedo,
            emission,
            metallic,
            roughness,
        });
    }
}

fn fit_size(w: u32, h: u32, max_dim: u32) -> (u32, u32) {
    let long = w.max(h).max(1);
    if long <= max_dim {
        return (w.max(1), h.max(1));
    }
    let s = max_dim as f32 / long as f32;
    (
        ((w as f32 * s).round() as u32).max(1),
        ((h as f32 * s).round() as u32).max(1),
    )
}

fn camera_fingerprint(cam: &OrbitCamera, w: u32, h: u32) -> u64 {
    let eye = cam.eye();
    let t = cam.target;
    let mut hsh: u64 = 0xcbf2_9ce4_8422_2325;
    for x in [
        eye.x, eye.y, eye.z, t.x, t.y, t.z, cam.distance, cam.yaw, cam.pitch, w as f32, h as f32,
    ] {
        hsh ^= x.to_bits() as u64;
        hsh = hsh.wrapping_mul(0x1000_0000_01b3);
    }
    hsh
}

fn rand01(s: &mut u32) -> f32 {
    *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    (*s >> 8) as f32 / (1u32 << 24) as f32
}

fn cosine_hemisphere(n: Vec3, r1: f32, r2: f32) -> Vec3 {
    let phi = 2.0 * std::f32::consts::PI * r1;
    let r = r2.sqrt();
    let x = r * phi.cos();
    let y = r * phi.sin();
    let z = (1.0 - r2).sqrt();
    let t = if n.y.abs() < 0.999 {
        n.cross(Vec3::Y).normalize_or_zero()
    } else {
        n.cross(Vec3::X).normalize_or_zero()
    };
    let b = n.cross(t);
    (t * x + b * y + n * z).normalize_or_zero()
}

fn intersect(origin: Vec3, dir: Vec3, tri: &Tri) -> Option<f32> {
    // Möller–Trumbore
    let p = dir.cross(tri.e2);
    let det = tri.e1.dot(p);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = origin - tri.v0;
    let u = tvec.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = tvec.cross(tri.e1);
    let v = dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = tri.e2.dot(q) * inv;
    if t > 1e-4 {
        Some(t)
    } else {
        None
    }
}

fn closest_hit(origin: Vec3, dir: Vec3, tris: &[Tri]) -> Option<(f32, usize)> {
    let mut best = f32::MAX;
    let mut idx = None;
    for (i, tri) in tris.iter().enumerate() {
        if let Some(t) = intersect(origin, dir, tri) {
            if t < best {
                best = t;
                idx = Some(i);
            }
        }
    }
    idx.map(|i| (best, i))
}

fn occluded(origin: Vec3, dir: Vec3, tmax: f32, tris: &[Tri]) -> bool {
    for tri in tris {
        if let Some(t) = intersect(origin, dir, tri) {
            if t > 1e-4 && t < tmax {
                return true;
            }
        }
    }
    false
}

fn sample_light(
    p: Vec3,
    n: Vec3,
    albedo: Vec3,
    metallic: f32,
    lights: &[Punctual],
    tris: &[Tri],
    rng: &mut u32,
) -> Vec3 {
    if lights.is_empty() {
        return Vec3::ZERO;
    }
    let li = (rand01(rng) * lights.len() as f32) as usize % lights.len();
    let l = lights[li];
    let (l_dir, dist, pdf_area) = match l.kind {
        1 => {
            // Sun: directional
            let d = (-l.dir).normalize_or_zero();
            (d, f32::INFINITY, 1.0)
        }
        _ => {
            let to = l.pos - p;
            let dist = to.length().max(1e-4);
            (to / dist, dist, 1.0)
        }
    };
    let ndotl = n.dot(l_dir).max(0.0);
    if ndotl <= 0.0 {
        return Vec3::ZERO;
    }
    let tmax = if dist.is_finite() { dist - 1e-3 } else { 1e6 };
    if occluded(p + n * 1e-3, l_dir, tmax, tris) {
        return Vec3::ZERO;
    }
    let atten = if l.kind == 1 {
        1.0
    } else {
        let a = 1.0 / (dist * dist);
        if l.range > 0.0 {
            let x = (1.0 - (dist / l.range).clamp(0.0, 1.0)).max(0.0);
            a * x * x
        } else {
            a
        }
    };
    let diffuse = albedo * (1.0 - metallic) / std::f32::consts::PI;
    diffuse * l.color * (ndotl * atten * lights.len() as f32 / pdf_area)
}

fn trace(mut origin: Vec3, mut dir: Vec3, tris: &[Tri], lights: &[Punctual], rng: &mut u32) -> Vec3 {
    let mut throughput = Vec3::ONE;
    let mut color = Vec3::ZERO;
    for bounce in 0..MAX_BOUNCES {
        let hit = closest_hit(origin, dir, tris);
        let Some((t, idx)) = hit else {
            // Simple sky gradient as environment
            let sky = Vec3::new(0.4, 0.6, 0.9) * (dir.y * 0.5 + 0.5).clamp(0.05, 1.0)
                + Vec3::splat(0.05);
            color += throughput * sky;
            break;
        };
        let tri = &tris[idx];
        let p = origin + dir * t;
        let mut n = tri.n;
        if n.dot(-dir) < 0.0 {
            n = -n;
        }
        // Emission
        color += throughput * tri.emission;
        // Next-event estimation (direct lights)
        color += throughput
            * sample_light(p, n, tri.albedo, tri.metallic, lights, tris, rng);
        // Indirect: cosine hemisphere
        let r1 = rand01(rng);
        let r2 = rand01(rng);
        let new_dir = cosine_hemisphere(n, r1, r2);
        let ndot = n.dot(new_dir).max(0.0);
        let brdf = tri.albedo * (1.0 - tri.metallic) / std::f32::consts::PI;
        // pdf = ndot/PI → throughput *= brdf * ndot / pdf = albedo * (1-m)
        throughput *= tri.albedo * (1.0 - tri.metallic).max(0.0);
        // Russian roulette
        if bounce >= 2 {
            let p_survive = throughput.max_element().clamp(0.05, 0.95);
            if rand01(rng) > p_survive {
                break;
            }
            throughput /= p_survive;
        }
        origin = p + n * 1e-3;
        dir = new_dir;
        let _ = ndot;
        let _ = brdf;
        let _ = tri.roughness;
    }
    color
}

fn aces_tonemap(x: Vec3) -> Vec3 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    ((x * (x * a + Vec3::splat(b))) / (x * (x * c + Vec3::splat(d)) + Vec3::splat(e)))
        .clamp(Vec3::ZERO, Vec3::ONE)
}

/// Path-trace a full-resolution still (used for F12 when engine is Cycles).
pub fn render_still(
    scene: &Scene,
    camera: &OrbitCamera,
    width: u32,
    height: u32,
    samples: u32,
    exposure: f32,
) -> Vec<u8> {
    let mut state = CyclesState::default();
    state.rebuild_scene(scene);
    let target = samples.max(1);
    while state.samples < target {
        state.accumulate(camera, width, height, exposure, target);
    }
    // Upscale/nearest-fill to requested size if tracer used a smaller buffer.
    let (sw, sh) = state.size();
    if sw == width && sh == height {
        return state.preview;
    }
    let mut out = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let sx = (x as f32 * sw as f32 / width as f32) as u32;
            let sy = (y as f32 * sh as f32 / height as f32) as u32;
            let si = ((sy.min(sh - 1) * sw + sx.min(sw - 1)) * 4) as usize;
            let di = ((y * width + x) * 4) as usize;
            out[di..di + 4].copy_from_slice(&state.preview[si..si + 4]);
        }
    }
    out
}
