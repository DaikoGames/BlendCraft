//! Object constraints, evaluated on the world matrix (after parenting).
use glam::{EulerRot, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ConstraintKind {
    CopyLocation {
        target: Option<u64>,
        axes: [bool; 3],
    },
    CopyRotation {
        target: Option<u64>,
    },
    CopyScale {
        target: Option<u64>,
    },
    CopyTransforms {
        target: Option<u64>,
    },
    /// Rotate so the local -Z axis points at the target (Y up). Great for cameras and lights.
    TrackTo {
        target: Option<u64>,
    },
    LimitLocation {
        min: [f32; 3],
        max: [f32; 3],
        enable: [bool; 3],
    },
    LimitRotation {
        min_deg: [f32; 3],
        max_deg: [f32; 3],
        enable: [bool; 3],
    },
    LimitScale {
        min: [f32; 3],
        max: [f32; 3],
        enable: [bool; 3],
    },
    LimitDistance {
        target: Option<u64>,
        distance: f32,
    },
    /// Keep the object above a world height (Y).
    Floor {
        height: f32,
    },
}

impl ConstraintKind {
    pub fn name(&self) -> &'static str {
        match self {
            ConstraintKind::CopyLocation { .. } => "Copy Location",
            ConstraintKind::CopyRotation { .. } => "Copy Rotation",
            ConstraintKind::CopyScale { .. } => "Copy Scale",
            ConstraintKind::CopyTransforms { .. } => "Copy Transforms",
            ConstraintKind::TrackTo { .. } => "Track To",
            ConstraintKind::LimitLocation { .. } => "Limit Location",
            ConstraintKind::LimitRotation { .. } => "Limit Rotation",
            ConstraintKind::LimitScale { .. } => "Limit Scale",
            ConstraintKind::LimitDistance { .. } => "Limit Distance",
            ConstraintKind::Floor { .. } => "Floor",
        }
    }

    pub fn target(&self) -> Option<u64> {
        match self {
            ConstraintKind::CopyLocation { target, .. }
            | ConstraintKind::CopyRotation { target }
            | ConstraintKind::CopyScale { target }
            | ConstraintKind::CopyTransforms { target }
            | ConstraintKind::TrackTo { target }
            | ConstraintKind::LimitDistance { target, .. } => *target,
            _ => None,
        }
    }

    pub fn target_mut(&mut self) -> Option<&mut Option<u64>> {
        match self {
            ConstraintKind::CopyLocation { target, .. }
            | ConstraintKind::CopyRotation { target }
            | ConstraintKind::CopyScale { target }
            | ConstraintKind::CopyTransforms { target }
            | ConstraintKind::TrackTo { target }
            | ConstraintKind::LimitDistance { target, .. } => Some(target),
            _ => None,
        }
    }

    /// One default instance of each constraint (for the Add Constraint menu).
    pub fn all_defaults() -> Vec<ConstraintKind> {
        vec![
            ConstraintKind::CopyLocation {
                target: None,
                axes: [true; 3],
            },
            ConstraintKind::CopyRotation { target: None },
            ConstraintKind::CopyScale { target: None },
            ConstraintKind::CopyTransforms { target: None },
            ConstraintKind::TrackTo { target: None },
            ConstraintKind::LimitLocation {
                min: [-10.0; 3],
                max: [10.0; 3],
                enable: [false; 3],
            },
            ConstraintKind::LimitRotation {
                min_deg: [-180.0; 3],
                max_deg: [180.0; 3],
                enable: [false; 3],
            },
            ConstraintKind::LimitScale {
                min: [0.1; 3],
                max: [10.0; 3],
                enable: [false; 3],
            },
            ConstraintKind::LimitDistance {
                target: None,
                distance: 5.0,
            },
            ConstraintKind::Floor { height: 0.0 },
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Constraint {
    pub enabled: bool,
    pub influence: f32,
    pub kind: ConstraintKind,
}

impl Constraint {
    pub fn new(kind: ConstraintKind) -> Self {
        Self {
            enabled: true,
            influence: 1.0,
            kind,
        }
    }

    /// `target` is the world matrix of the target object (None if unset / missing).
    pub fn apply(&self, world: Mat4, target: Option<Mat4>) -> Mat4 {
        let (s, r, t) = world.to_scale_rotation_translation();
        let compose = |s: Vec3, r: Quat, t: Vec3| Mat4::from_scale_rotation_translation(s, r, t);
        let res = match &self.kind {
            ConstraintKind::CopyLocation { axes, .. } => {
                let Some(tw) = target else { return world };
                let tt = tw.w_axis.truncate();
                let mut nt = t;
                for k in 0..3 {
                    if axes[k] {
                        nt[k] = tt[k];
                    }
                }
                compose(s, r, nt)
            }
            ConstraintKind::CopyRotation { .. } => {
                let Some(tw) = target else { return world };
                let (_, tr, _) = tw.to_scale_rotation_translation();
                compose(s, tr, t)
            }
            ConstraintKind::CopyScale { .. } => {
                let Some(tw) = target else { return world };
                let (ts, _, _) = tw.to_scale_rotation_translation();
                compose(ts, r, t)
            }
            ConstraintKind::CopyTransforms { .. } => match target {
                Some(tw) => tw,
                None => return world,
            },
            ConstraintKind::TrackTo { .. } => {
                let Some(tw) = target else { return world };
                let tt = tw.w_axis.truncate();
                let dir = tt - t;
                if dir.length() < 1e-6 {
                    return world;
                }
                let up = if dir.normalize().cross(Vec3::Y).length() < 1e-3 {
                    Vec3::Z
                } else {
                    Vec3::Y
                };
                let rot = Quat::from_mat4(&Mat4::look_at_rh(t, tt, up).inverse());
                compose(s, rot, t)
            }
            ConstraintKind::LimitLocation { min, max, enable } => {
                let mut nt = t;
                for k in 0..3 {
                    if enable[k] {
                        nt[k] = nt[k].clamp(min[k].min(max[k]), max[k].max(min[k]));
                    }
                }
                compose(s, r, nt)
            }
            ConstraintKind::LimitRotation {
                min_deg,
                max_deg,
                enable,
            } => {
                let (x, y, z) = r.to_euler(EulerRot::XYZ);
                let mut e = [x, y, z];
                for k in 0..3 {
                    if enable[k] {
                        let (a, b) = (min_deg[k].to_radians(), max_deg[k].to_radians());
                        e[k] = e[k].clamp(a.min(b), b.max(a));
                    }
                }
                compose(s, Quat::from_euler(EulerRot::XYZ, e[0], e[1], e[2]), t)
            }
            ConstraintKind::LimitScale { min, max, enable } => {
                let mut ns = s;
                for k in 0..3 {
                    if enable[k] {
                        ns[k] = ns[k].clamp(min[k].min(max[k]), max[k].max(min[k]));
                    }
                }
                compose(ns, r, t)
            }
            ConstraintKind::LimitDistance { distance, .. } => {
                let Some(tw) = target else { return world };
                let tt = tw.w_axis.truncate();
                let d = t - tt;
                if d.length() > *distance && d.length() > 1e-6 {
                    compose(s, r, tt + d.normalize() * *distance)
                } else {
                    return world;
                }
            }
            ConstraintKind::Floor { height } => {
                let mut nt = t;
                nt.y = nt.y.max(*height);
                compose(s, r, nt)
            }
        };
        if (self.influence - 1.0).abs() < 1e-4 {
            return res;
        }
        let w = self.influence.clamp(0.0, 1.0);
        let (s1, r1, t1) = res.to_scale_rotation_translation();
        compose(s.lerp(s1, w), r.slerp(r1, w), t.lerp(t1, w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_clamps_height() {
        let c = Constraint::new(ConstraintKind::Floor { height: 0.0 });
        let m = c.apply(Mat4::from_translation(Vec3::new(1.0, -3.0, 2.0)), None);
        assert!((m.w_axis.y - 0.0).abs() < 1e-5);
        assert!((m.w_axis.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn copy_location_follows_target() {
        let c = Constraint::new(ConstraintKind::CopyLocation {
            target: Some(1),
            axes: [true, false, true],
        });
        let m = c.apply(
            Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0)),
            Some(Mat4::from_translation(Vec3::new(2.0, 9.0, 3.0))),
        );
        assert!(
            (m.w_axis.x - 2.0).abs() < 1e-5
                && (m.w_axis.y - 5.0).abs() < 1e-5
                && (m.w_axis.z - 3.0).abs() < 1e-5
        );
    }
}
