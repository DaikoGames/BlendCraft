/// How an object is outlined in the viewport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Highlight {
    None,
    Selected,
    Active,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadingMode {
    Rendered = 0,
    Albedo = 1,
    Normals = 2,
    Roughness = 3,
    Metallic = 4,
    UvChecker = 5,
}

impl ShadingMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Rendered => "Rendered",
            Self::Albedo => "Albedo",
            Self::Normals => "Normals",
            Self::Roughness => "Roughness",
            Self::Metallic => "Metallic",
            Self::UvChecker => "UV Checker",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireMode {
    Off,
    Overlay,
    Only,
}

impl WireMode {
    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::Overlay,
            Self::Overlay => Self::Only,
            Self::Only => Self::Off,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Overlay => "overlay",
            Self::Only => "only",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ViewSettings {
    pub shading: ShadingMode,
    pub wireframe: WireMode,
    pub show_background: bool,
    /// 0 = sharp, 1 = fully blurred environment background
    pub background_blur: f32,
    pub exposure: f32,
    /// multiplier for image-based lighting (turn down to let scene lights dominate)
    pub env_strength: f32,
    /// radians around Y
    pub env_rotation: f32,
    /// Real-time directional/spot shadow mapping (first suitable light).
    pub shadows: bool,
    /// Softness of the shadow PCF filter (0 = hard, higher = softer).
    pub shadow_softness: f32,
}

impl Default for ViewSettings {
    fn default() -> Self {
        Self {
            shading: ShadingMode::Rendered,
            wireframe: WireMode::Off,
            // Off by default so the viewport behaves like a pure 3D engine / Blender solid mode.
            // User can enable the procedural sky (or load an HDRI) from the View menu.
            show_background: false,
            background_blur: 0.35,
            exposure: 1.0,
            env_strength: 0.0, // no IBL contribution until the user enables a sky/HDRI
            env_rotation: 0.0,
            shadows: true,
            shadow_softness: 1.5,
        }
    }
}
