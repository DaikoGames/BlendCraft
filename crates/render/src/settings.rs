/// How an object is outlined in the viewport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Highlight {
    None,
    Selected,
    Active,
}

/// Blender-style render engines. These drive different lighting models in the viewport
/// and offline capture path (Workbench = solid studio, EEVEE = real-time PBR,
/// Cycles = multi-sample path-traced look).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RenderEngine {
    /// Solid / studio viewport — fast, no IBL, three-point lighting.
    Workbench = 0,
    /// Real-time PBR with shadows and optional image-based lighting.
    #[default]
    Eevee = 1,
    /// Path-traced style: multi-sampled environment + soft direct lights.
    Cycles = 2,
}

impl RenderEngine {
    pub fn name(self) -> &'static str {
        match self {
            Self::Workbench => "Workbench",
            Self::Eevee => "EEVEE",
            Self::Cycles => "Cycles",
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Workbench,
            2 => Self::Cycles,
            _ => Self::Eevee,
        }
    }
}

/// Blender 3D Viewport header shading buttons.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ViewportShading {
    Wireframe,
    #[default]
    Solid,
    MaterialPreview,
    Rendered,
}

impl ViewportShading {
    pub fn name(self) -> &'static str {
        match self {
            Self::Wireframe => "Wireframe",
            Self::Solid => "Solid",
            Self::MaterialPreview => "Material Preview",
            Self::Rendered => "Rendered",
        }
    }
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
    /// Blender viewport shading (Wireframe / Solid / Material / Rendered).
    pub viewport_shading: ViewportShading,
    pub wireframe: WireMode,
    /// See-through mesh display (Blender X-Ray).
    pub xray: bool,
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
    /// Active render engine (Workbench / EEVEE / Cycles).
    pub engine: RenderEngine,
    /// Cycles path-trace sample target for the interactive viewport.
    pub cycles_samples: u32,
    /// Cap on the long edge of the Cycles viewport buffer (pixels). Higher = sharper, slower.
    pub cycles_max_dim: u32,
    /// Viewport pixel size (1 = full resolution, 2 = half, 4 = quarter). Like Blender's pixel size.
    pub cycles_pixel_size: u32,
    /// Size (px) of the Cycles viewport buffer, derived from the Output resolution
    /// (X × Y × %). `[0, 0]` = follow the panel size. Pixel Size divides this further.
    pub cycles_view_size: [u32; 2],
}

impl Default for ViewSettings {
    fn default() -> Self {
        Self {
            shading: ShadingMode::Rendered,
            viewport_shading: ViewportShading::Solid,
            wireframe: WireMode::Off,
            xray: false,
            show_background: false,
            background_blur: 0.35,
            exposure: 1.0,
            env_strength: 0.0,
            env_rotation: 0.0,
            shadows: true,
            shadow_softness: 1.5,
            engine: RenderEngine::Eevee,
            cycles_samples: 64,
            cycles_max_dim: 1280,
            cycles_pixel_size: 1,
            cycles_view_size: [0, 0],
        }
    }
}
