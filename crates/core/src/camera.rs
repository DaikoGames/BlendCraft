use crate::mesh::Aabb;
use glam::{Mat4, Vec3};

/// A fixed view (looking through a camera object). While set, the orbit controls are disabled.
#[derive(Clone, Debug)]
pub struct FixedView {
    pub eye: Vec3,
    pub view: Mat4,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    /// Some(height) = orthographic
    pub ortho_height: Option<f32>,
}

/// Orbit camera, Y-up, right-handed.
#[derive(Clone, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    pub fixed: Option<FixedView>,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 5.0,
            yaw: 0.6,
            pitch: 0.4,
            fov_y: 45f32.to_radians(),
            near: 0.05,
            far: 2000.0,
            fixed: None,
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        if let Some(f) = &self.fixed {
            return f.eye;
        }
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + self.distance * Vec3::new(cp * sy, sp, cp * cy)
    }

    pub fn view(&self) -> Mat4 {
        if let Some(f) = &self.fixed {
            return f.view;
        }
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let aspect = aspect.max(0.01);
        if let Some(f) = &self.fixed {
            let proj = match f.ortho_height {
                Some(h) => Mat4::orthographic_rh(
                    -h * aspect * 0.5,
                    h * aspect * 0.5,
                    -h * 0.5,
                    h * 0.5,
                    f.near,
                    f.far,
                ),
                None => Mat4::perspective_rh(f.fov_y, aspect, f.near, f.far),
            };
            return proj * f.view;
        }
        Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far) * self.view()
    }

    /// World-space ray through a point in normalized device coordinates (x right, y up, both -1..1).
    pub fn ray(&self, ndc: (f32, f32), aspect: f32) -> (Vec3, Vec3) {
        let inv = self.view_proj(aspect).inverse();
        let p0 = inv.project_point3(Vec3::new(ndc.0, ndc.1, 0.0));
        let p1 = inv.project_point3(Vec3::new(ndc.0, ndc.1, 1.0));
        (p0, (p1 - p0).normalize())
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        if self.fixed.is_some() {
            return;
        }
        self.yaw -= dx * 0.005;
        self.pitch = (self.pitch + dy * 0.005).clamp(-1.55, 1.55);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        if self.fixed.is_some() {
            return;
        }
        let v = self.view().inverse();
        let (right, up) = (v.x_axis.truncate(), v.y_axis.truncate());
        let s = self.distance * 0.0015;
        self.target += (-right * dx + up * dy) * s;
    }

    pub fn zoom(&mut self, amount: f32) {
        if self.fixed.is_some() {
            return;
        }
        self.distance = (self.distance * (1.0 - amount * 0.1)).clamp(0.01, 1500.0);
    }

    pub fn frame(&mut self, b: Aabb) {
        if self.fixed.is_some() {
            return;
        }
        self.target = b.center();
        let r = b.radius().max(0.001);
        self.distance = r / (self.fov_y * 0.5).sin() * 1.1;
        self.near = (self.distance * 0.01).max(0.001);
        self.far = self.distance * 100.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eye_is_distance_from_target() {
        let c = OrbitCamera::default();
        assert!(((c.eye() - c.target).length() - c.distance).abs() < 1e-4);
    }
}
