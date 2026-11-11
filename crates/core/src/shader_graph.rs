//! Serializable material shader graph and CPU evaluation into PBR slots / textures.
use crate::material::{Material, TextureData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShaderNodeKind {
    Output,
    LightOutput,
    WorldOutput,
    AovOutput,
    Principled,
    ImageTexture,
    Rgb,
    TexCoord,
    Value,
    Mapping,
    Noise,
    Voronoi,
    ColorRamp,
    Mix,
    Bump,
    NormalMap,
    Emission,
    Glossy,
    Diffuse,
    // ---- Vector / Utilities (Blender Shader Editor) ----
    CombineXyz,
    SeparateXyz,
    VectorMath,
    VectorRotate,
    VectorTransform,
    VectorCurves,
    Normal,
    Displacement,
    VectorDisplacement,
    Math,
    MapRange,
    Clamp,
    FloatCurve,
    Switch,
    IndexSwitch,
    MenuSwitch,
    Script,
    Group,
    // ---- Color category (Blender Shader Editor) ----
    Blackbody,
    BrightContrast,
    Gamma,
    HueSatVal,
    InvertColor,
    LightFalloff,
    RgbCurves,
    Wavelength,
    CombineColor,
    SeparateColor,
    RgbToBw,
    ShaderToRgb,
    // ---- Texture category (Blender Shader Editor) ----
    Brick,
    Checker,
    Environment,
    Gabor,
    Gradient,
    Ies,
    Magic,
    Sky,
    Wave,
    WhiteNoise,
    // ---- Shader category (Blender Shader Editor) ----
    AddShader,
    Background,
    GlassBsdf,
    HairBsdf,
    Holdout,
    LightAccumulation,
    MixShader,
    MetallicBsdf,
    PrincipledHair,
    PrincipledVolume,
    RayPortalBsdf,
    RefractionBsdf,
    SpecularBsdf,
    SubsurfaceScattering,
    ToonBsdf,
    TranslucentBsdf,
    TransparentBsdf,
    SheenBsdf,
    VolumeAbsorption,
    VolumeScatter,
    VolumeCoefficients,
    // ---- Input (Blender Shader Editor) ----
    AmbientOcclusion,
    Attribute,
    Bevel,
    CameraData,
    Fresnel,
    Geometry,
    CurvesInfo,
    LayerWeight,
    LightEvaluation,
    LightInfo,
    LightPath,
    ObjectInfo,
    ParticleInfo,
    PointInfo,
    Raycast,
    SceneTime,
    ShadowRaycast,
    Tangent,
    UvMap,
    ColorAttribute,
    VolumeInfo,
    Wireframe,
}

impl ShaderNodeKind {
    pub fn title(self) -> &'static str {
        match self {
            Self::Output => "Material Output",
            Self::LightOutput => "Light Output",
            Self::WorldOutput => "World Output",
            Self::AovOutput => "AOV Output",
            Self::Principled => "Principled BSDF",
            Self::ImageTexture => "Image Texture",
            Self::Rgb => "RGB",
            Self::TexCoord => "Texture Coordinate",
            Self::Value => "Value",
            Self::Mapping => "Mapping",
            Self::Noise => "Noise Texture",
            Self::Voronoi => "Voronoi Texture",
            Self::ColorRamp => "Color Ramp",
            Self::Mix => "Mix",
            Self::Bump => "Bump",
            Self::NormalMap => "Normal Map",
            Self::Emission => "Emission",
            Self::Glossy => "Glossy BSDF",
            Self::Diffuse => "Diffuse BSDF",
            Self::CombineXyz => "Combine XYZ",
            Self::SeparateXyz => "Separate XYZ",
            Self::VectorMath => "Vector Math",
            Self::VectorRotate => "Vector Rotate",
            Self::VectorTransform => "Vector Transform",
            Self::VectorCurves => "Vector Curves",
            Self::Normal => "Normal",
            Self::Displacement => "Displacement",
            Self::VectorDisplacement => "Vector Displacement",
            Self::Math => "Math",
            Self::MapRange => "Map Range",
            Self::Clamp => "Clamp",
            Self::FloatCurve => "Float Curve",
            Self::Switch => "Switch",
            Self::IndexSwitch => "Index Switch",
            Self::MenuSwitch => "Menu Switch",
            Self::Script => "Script",
            Self::Group => "Group",
            Self::Blackbody => "Blackbody",
            Self::BrightContrast => "Brightness/Contrast",
            Self::Gamma => "Gamma",
            Self::HueSatVal => "Hue/Saturation/Value",
            Self::InvertColor => "Invert Color",
            Self::LightFalloff => "Light Falloff",
            Self::RgbCurves => "RGB Curves",
            Self::Wavelength => "Wavelength",
            Self::CombineColor => "Combine Color",
            Self::SeparateColor => "Separate Color",
            Self::RgbToBw => "RGB to BW",
            Self::ShaderToRgb => "Shader to RGB",
            Self::Brick => "Brick Texture",
            Self::Checker => "Checker Texture",
            Self::Environment => "Environment Texture",
            Self::Gabor => "Gabor Texture",
            Self::Gradient => "Gradient Texture",
            Self::Ies => "IES Texture",
            Self::Magic => "Magic Texture",
            Self::Sky => "Sky Texture",
            Self::Wave => "Wave Texture",
            Self::WhiteNoise => "White Noise Texture",
            Self::AddShader => "Add Shader",
            Self::Background => "Background",
            Self::GlassBsdf => "Glass BSDF",
            Self::HairBsdf => "Hair BSDF",
            Self::Holdout => "Holdout",
            Self::LightAccumulation => "Light Accumulation",
            Self::MixShader => "Mix Shader",
            Self::MetallicBsdf => "Metallic BSDF",
            Self::PrincipledHair => "Principled Hair BSDF",
            Self::PrincipledVolume => "Principled Volume",
            Self::RayPortalBsdf => "Ray Portal BSDF",
            Self::RefractionBsdf => "Refraction BSDF",
            Self::SpecularBsdf => "Specular BSDF",
            Self::SubsurfaceScattering => "Subsurface Scattering",
            Self::ToonBsdf => "Toon BSDF",
            Self::TranslucentBsdf => "Translucent BSDF",
            Self::TransparentBsdf => "Transparent BSDF",
            Self::SheenBsdf => "Sheen BSDF",
            Self::VolumeAbsorption => "Volume Absorption",
            Self::VolumeScatter => "Volume Scatter",
            Self::VolumeCoefficients => "Volume Coefficients",
            Self::AmbientOcclusion => "Ambient Occlusion",
            Self::Attribute => "Attribute",
            Self::Bevel => "Bevel",
            Self::CameraData => "Camera Data",
            Self::Fresnel => "Fresnel",
            Self::Geometry => "Geometry",
            Self::CurvesInfo => "Curves Info",
            Self::LayerWeight => "Layer Weight",
            Self::LightEvaluation => "Light Evaluation",
            Self::LightInfo => "Light Info",
            Self::LightPath => "Light Path",
            Self::ObjectInfo => "Object Info",
            Self::ParticleInfo => "Particle Info",
            Self::PointInfo => "Point Info",
            Self::Raycast => "Raycast",
            Self::SceneTime => "Scene Time",
            Self::ShadowRaycast => "Shadow Raycast",
            Self::Tangent => "Tangent",
            Self::UvMap => "UV Map",
            Self::ColorAttribute => "Color Attribute",
            Self::VolumeInfo => "Volume Info",
            Self::Wireframe => "Wireframe",
        }
    }

    /// True for the Vector / Utilities nodes (Combine XYZ … Group).
    pub fn is_vector_util(self) -> bool {
        matches!(
            self,
            Self::CombineXyz
                | Self::SeparateXyz
                | Self::VectorMath
                | Self::VectorRotate
                | Self::VectorTransform
                | Self::VectorCurves
                | Self::Normal
                | Self::Displacement
                | Self::VectorDisplacement
                | Self::Math
                | Self::MapRange
                | Self::Clamp
                | Self::FloatCurve
                | Self::Switch
                | Self::IndexSwitch
                | Self::MenuSwitch
                | Self::Script
                | Self::Group
        )
    }

    /// Input socket labels (socket order) of the Vector / Utilities nodes.
    pub fn vu_inputs(self) -> &'static [&'static str] {
        match self {
            Self::CombineXyz => &["X", "Y", "Z"],
            Self::SeparateXyz => &["Vector"],
            Self::VectorMath => &["A", "B", "Scale"],
            Self::VectorRotate => &["Vector", "Angle"],
            Self::VectorTransform | Self::VectorCurves => &["Vector"],
            Self::Normal => &["Normal"],
            Self::Displacement => &["Height", "Midlevel", "Scale"],
            Self::VectorDisplacement => &["Vector", "Midlevel", "Scale"],
            Self::Math => &["A", "B", "C"],
            Self::MapRange => &["Value", "From Min", "From Max", "To Min", "To Max"],
            Self::Clamp => &["Value", "Min", "Max"],
            Self::FloatCurve => &["Fac", "Value"],
            Self::Switch => &["Switch", "False", "True"],
            Self::IndexSwitch => &["Index", "0", "1", "2", "3"],
            Self::MenuSwitch => &["A", "B", "C", "D"],
            Self::Script => &["a", "b", "c"],
            Self::Group => &["Input 1", "Input 2", "Input 3"],
            _ => &[],
        }
    }

    /// Output socket labels (socket order) of the Vector / Utilities nodes.
    pub fn vu_outputs(self) -> &'static [&'static str] {
        match self {
            Self::CombineXyz | Self::VectorRotate | Self::VectorTransform | Self::VectorCurves => &["Vector"],
            Self::SeparateXyz => &["X", "Y", "Z"],
            Self::VectorMath => &["Vector", "Value"],
            Self::Normal => &["Normal", "Dot"],
            Self::Displacement | Self::VectorDisplacement => &["Displacement"],
            Self::Math | Self::MapRange | Self::Clamp => &["Result"],
            Self::FloatCurve => &["Value"],
            Self::Switch | Self::IndexSwitch | Self::MenuSwitch => &["Output"],
            Self::Script => &["Value", "Color"],
            Self::Group => &["Output 1", "Output 2", "Output 3"],
            _ => &[],
        }
    }

    /// Editable number rows: (label, slot). Slots 0–3 = `params`, 4–6 = `color` x/y/z, 7 = `value`.
    pub fn vu_rows(self) -> &'static [(&'static str, u8)] {
        match self {
            Self::CombineXyz => &[("X", 0), ("Y", 1), ("Z", 2)],
            Self::SeparateXyz => &[("X", 4), ("Y", 5), ("Z", 6)],
            Self::VectorMath => &[
                ("A.x", 4), ("A.y", 5), ("A.z", 6), ("B.x", 0), ("B.y", 1), ("B.z", 2), ("Scale", 3),
            ],
            Self::VectorRotate => &[("X", 4), ("Y", 5), ("Z", 6), ("Angle", 0)],
            Self::VectorTransform => &[("X", 4), ("Y", 5), ("Z", 6)],
            Self::VectorCurves => &[
                ("X", 4), ("Y", 5), ("Z", 6), ("Curve X", 0), ("Curve Y", 1), ("Curve Z", 2), ("Fac", 3),
            ],
            Self::Normal => &[("X", 0), ("Y", 1), ("Z", 2)],
            Self::Displacement => &[("Height", 0), ("Midlevel", 1), ("Scale", 2)],
            Self::VectorDisplacement => &[("X", 4), ("Y", 5), ("Z", 6), ("Midlevel", 0), ("Scale", 1)],
            Self::Math => &[("A", 0), ("B", 1), ("C", 2)],
            Self::MapRange => &[("Value", 7), ("From Min", 0), ("From Max", 1), ("To Min", 2), ("To Max", 3)],
            Self::Clamp => &[("Value", 7), ("Min", 0), ("Max", 1)],
            Self::FloatCurve => &[("Fac", 0), ("Curve", 1), ("Value", 7)],
            Self::IndexSwitch => &[("Index", 0)],
            Self::Script => &[("a", 0), ("b", 1), ("c", 2)],
            _ => &[],
        }
    }

    /// Drop-down choices: (label, slot, option names). Slot numbering as in `vu_rows`.
    pub fn vu_choices(self) -> &'static [(&'static str, u8, &'static [&'static str])] {
        match self {
            Self::Math => &[("Operation", 7, MATH_OPS)],
            Self::VectorMath => &[("Operation", 7, VECTOR_MATH_OPS)],
            Self::VectorRotate => &[("Axis", 7, &["X", "Y", "Z"])],
            Self::VectorTransform => &[
                ("Type", 7, &["Point", "Vector", "Normal"]),
                ("From", 0, &["World", "Object", "Camera"]),
                ("To", 1, &["World", "Object", "Camera"]),
            ],
            Self::Switch => &[("Switch", 0, &["False", "True"])],
            Self::MenuSwitch => &[("Menu", 0, &["A", "B", "C", "D"])],
            _ => &[],
        }
    }

    /// Initial text (Script expression / Group name).
    pub fn default_text(self) -> &'static str {
        match self {
            Self::Script => "sin(u * 6.2832 * a) * 0.5 + 0.5",
            Self::Group => "NodeGroup",
            _ => "",
        }
    }

    /// Initial `color` (its xyz double as the default input vector of Vector nodes).
    pub fn default_color(self) -> [f32; 4] {
        if self.is_vector_util() {
            [0.0, 0.0, 0.0, 1.0]
        } else {
            [0.8, 0.8, 0.8, 1.0]
        }
    }

    /// True for the Color-category nodes added beyond Color Ramp / Mix.
    pub fn is_color_extra(self) -> bool {
        matches!(
            self,
            Self::Blackbody
                | Self::BrightContrast
                | Self::Gamma
                | Self::HueSatVal
                | Self::InvertColor
                | Self::LightFalloff
                | Self::RgbCurves
                | Self::Wavelength
                | Self::CombineColor
                | Self::SeparateColor
                | Self::RgbToBw
                | Self::ShaderToRgb
        )
    }

    /// Labels of the extra per-node parameters (`ShaderNode::params`). `mode` is the
    /// node's mode (0 = RGB, 1 = HSV, 2 = HSL) and only matters for Combine / Separate Color.
    pub fn param_labels(self, mode: f32) -> &'static [&'static str] {
        match self {
            Self::BrightContrast => &["Bright", "Contrast"],
            Self::HueSatVal => &["Hue", "Saturation", "Value", "Fac"],
            Self::LightFalloff => &["Smooth"],
            Self::RgbCurves => &["Combined", "Red", "Green", "Blue"],
            Self::CombineColor | Self::SeparateColor => match mode.round() as i32 {
                1 => &["Hue", "Saturation", "Value"],
                2 => &["Hue", "Saturation", "Lightness"],
                _ => &["Red", "Green", "Blue"],
            },
            _ => &[],
        }
    }

    pub fn param_range(self, index: usize) -> (f32, f32) {
        match self {
            Self::BrightContrast => (-1.0, 1.0),
            Self::HueSatVal => {
                if index == 1 || index == 2 {
                    (0.0, 2.0)
                } else {
                    (0.0, 1.0)
                }
            }
            Self::LightFalloff => (0.0, 10.0),
            _ => (0.0, 1.0),
        }
    }

    pub fn default_params(self) -> [f32; 4] {
        match self {
            Self::HueSatVal => [0.5, 1.0, 1.0, 1.0],
            Self::RgbCurves => [0.5, 0.5, 0.5, 0.5],
            Self::VectorMath => [0.0, 0.0, 0.0, 1.0],
            Self::VectorCurves => [0.5, 0.5, 0.5, 1.0],
            Self::Normal => [0.0, 0.0, 1.0, 0.0],
            Self::Displacement => [0.0, 0.5, 1.0, 0.0],
            Self::VectorDisplacement => [0.0, 1.0, 0.0, 0.0],
            Self::Math => [0.5, 0.5, 0.5, 0.0],
            Self::MapRange => [0.0, 1.0, 0.0, 1.0],
            Self::Clamp => [0.0, 1.0, 0.0, 0.0],
            Self::FloatCurve => [1.0, 0.5, 0.0, 0.0],
            Self::Script => [1.0, 1.0, 1.0, 0.0],
            _ => [0.0; 4],
        }
    }

    /// Output socket labels (in socket order) of the extra color nodes.
    pub fn color_outputs(self, mode: f32) -> &'static [&'static str] {
        match self {
            Self::LightFalloff => &["Quadratic", "Linear", "Constant"],
            Self::SeparateColor => self.param_labels(mode),
            Self::RgbToBw => &["Val"],
            Self::ShaderToRgb => &["Color", "Alpha"],
            _ => &["Color"],
        }
    }

    /// True for the Texture-category nodes added beyond Image / Noise / Voronoi.
    pub fn is_texture_extra(self) -> bool {
        matches!(
            self,
            Self::Brick
                | Self::Checker
                | Self::Environment
                | Self::Gabor
                | Self::Gradient
                | Self::Ies
                | Self::Magic
                | Self::Sky
                | Self::Wave
                | Self::WhiteNoise
        )
    }

    /// True for texture nodes whose `value` is a Scale.
    pub fn has_texture_scale(self) -> bool {
        matches!(
            self,
            Self::Brick | Self::Checker | Self::Gabor | Self::Ies | Self::Magic | Self::Wave
        )
    }

    /// Texture nodes that also expose a scalar Fac output (socket 1).
    pub fn has_fac_output(self) -> bool {
        self.is_texture_extra() && !matches!(self, Self::Environment | Self::Sky)
    }

    /// True for the Shader-category nodes added beyond Principled / Diffuse / Glossy / Emission.
    pub fn is_shader_extra(self) -> bool {
        matches!(
            self,
            Self::AddShader
                | Self::Background
                | Self::GlassBsdf
                | Self::HairBsdf
                | Self::Holdout
                | Self::LightAccumulation
                | Self::MixShader
                | Self::MetallicBsdf
                | Self::PrincipledHair
                | Self::PrincipledVolume
                | Self::RayPortalBsdf
                | Self::RefractionBsdf
                | Self::SpecularBsdf
                | Self::SubsurfaceScattering
                | Self::ToonBsdf
                | Self::TranslucentBsdf
                | Self::TransparentBsdf
                | Self::SheenBsdf
                | Self::VolumeAbsorption
                | Self::VolumeScatter
                | Self::VolumeCoefficients
        )
    }

    /// True for shader nodes that produce a volume (plug into Volume sockets).
    pub fn is_volume_shader(self) -> bool {
        matches!(
            self,
            Self::PrincipledVolume | Self::VolumeAbsorption | Self::VolumeScatter | Self::VolumeCoefficients
        )
    }

    /// Input socket labels (in socket order) for the extra shader nodes.
    pub fn extra_inputs(self) -> &'static [&'static str] {
        match self {
            Self::AddShader => &["Shader", "Shader"],
            Self::MixShader => &["Fac", "Shader", "Shader"],
            Self::Holdout | Self::LightAccumulation => &[],
            Self::Blackbody => &["Temperature"],
            Self::Wavelength => &["Wavelength"],
            Self::LightFalloff => &["Strength", "Smooth"],
            Self::CombineColor => &["Red", "Green", "Blue"],
            Self::ShaderToRgb => &["Shader"],
            Self::VolumeCoefficients => &["Absorption", "Scatter"],
            _ => &["Color"],
        }
    }

    /// Label of the node's editable scalar (`ShaderNode::value`), if it has one.
    pub fn value_label(self) -> Option<&'static str> {
        match self {
            Self::Background => Some("Strength"),
            Self::MixShader => Some("Fac"),
            Self::GlassBsdf
            | Self::HairBsdf
            | Self::MetallicBsdf
            | Self::PrincipledHair
            | Self::RefractionBsdf
            | Self::SpecularBsdf
            | Self::SheenBsdf => Some("Roughness"),
            Self::Brick | Self::Checker | Self::Gabor | Self::Ies | Self::Magic | Self::Wave => Some("Scale"),
            Self::SubsurfaceScattering => Some("Scale"),
            Self::ToonBsdf => Some("Smooth"),
            Self::PrincipledVolume
            | Self::VolumeAbsorption
            | Self::VolumeScatter
            | Self::VolumeCoefficients => Some("Density"),
            Self::Blackbody => Some("Temperature"),
            Self::Wavelength => Some("Wavelength"),
            Self::Gamma => Some("Gamma"),
            Self::InvertColor => Some("Fac"),
            Self::LightFalloff => Some("Strength"),
            _ => None,
        }
    }

    pub fn value_range(self) -> (f32, f32) {
        match self {
            Self::Background => (0.0, 10.0),
            Self::PrincipledVolume
            | Self::VolumeAbsorption
            | Self::VolumeScatter
            | Self::VolumeCoefficients => (0.0, 10.0),
            Self::Brick | Self::Checker | Self::Gabor | Self::Ies | Self::Magic | Self::Wave => (0.1, 50.0),
            Self::Blackbody => (800.0, 12000.0),
            Self::Wavelength => (380.0, 780.0),
            Self::Gamma => (0.1, 5.0),
            Self::LightFalloff => (0.0, 1000.0),
            _ => (0.0, 1.0),
        }
    }

    /// Initial value for `ShaderNode::value` when the node is added from the menu.
    pub fn default_value(self) -> f32 {
        match self {
            Self::Background
            | Self::PrincipledVolume
            | Self::VolumeAbsorption
            | Self::VolumeScatter
            | Self::VolumeCoefficients => 1.0,
            Self::Brick | Self::Checker | Self::Gabor | Self::Ies | Self::Magic | Self::Wave => 5.0,
            Self::Blackbody => 6500.0,
            Self::Wavelength => 550.0,
            Self::Gamma | Self::InvertColor => 1.0,
            Self::LightFalloff => 100.0,
            Self::CombineColor
            | Self::SeparateColor
            | Self::Math
            | Self::VectorMath
            | Self::VectorRotate
            | Self::VectorTransform => 0.0,
            _ => 0.5,
        }
    }

    /// Label for the primary output socket of the extra shader nodes.
    pub fn output_label(self) -> &'static str {
        match self {
            Self::Background => "Background ◉",
            Self::AddShader | Self::MixShader | Self::LightAccumulation => "Shader ◉",
            Self::PrincipledVolume
            | Self::VolumeAbsorption
            | Self::VolumeScatter
            | Self::VolumeCoefficients => "Volume ◉",
            _ => "BSDF ◉",
        }
    }

    /// True for any of the Output-category nodes (Material / Light / World / AOV).
    pub fn is_output(self) -> bool {
        matches!(self, Self::Output | Self::LightOutput | Self::WorldOutput | Self::AovOutput)
    }

    /// True for Blender-style Input category nodes (excluding RGB/Value/TexCoord handled above).
    pub fn is_input_extra(self) -> bool {
        matches!(
            self,
            Self::AmbientOcclusion
                | Self::Attribute
                | Self::Bevel
                | Self::CameraData
                | Self::Fresnel
                | Self::Geometry
                | Self::CurvesInfo
                | Self::LayerWeight
                | Self::LightEvaluation
                | Self::LightInfo
                | Self::LightPath
                | Self::ObjectInfo
                | Self::ParticleInfo
                | Self::PointInfo
                | Self::Raycast
                | Self::SceneTime
                | Self::ShadowRaycast
                | Self::Tangent
                | Self::UvMap
                | Self::ColorAttribute
                | Self::VolumeInfo
                | Self::Wireframe
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShaderNode {
    pub id: u64,
    pub kind: ShaderNodeKind,
    pub pos: [f32; 2],
    pub image_index: Option<usize>,
    pub value: f32,
    /// RGBA 0–1 for RGB / Emission color nodes.
    pub color: [f32; 4],
    /// Extra per-node parameters (see `ShaderNodeKind::param_labels`).
    #[serde(default)]
    pub params: [f32; 4],
    /// Script expression / Group name.
    #[serde(default)]
    pub text: String,
}

impl Default for ShaderNode {
    fn default() -> Self {
        Self {
            id: 0,
            kind: ShaderNodeKind::Value,
            pos: [0.0, 0.0],
            image_index: None,
            value: 0.5,
            color: [0.8, 0.8, 0.8, 1.0],
            params: [0.0; 4],
            text: String::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShaderLink {
    pub from: u64,
    pub to: u64,
    /// Optional socket names; empty = default primary sockets.
    #[serde(default)]
    pub from_socket: u8,
    #[serde(default)]
    pub to_socket: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShaderGraph {
    pub nodes: Vec<ShaderNode>,
    pub links: Vec<ShaderLink>,
    pub next_id: u64,
}

impl Default for ShaderGraph {
    fn default() -> Self {
        Self {
            nodes: vec![
                ShaderNode {
                    id: 1,
                    kind: ShaderNodeKind::Principled,
                    pos: [330.0, 70.0],
                    ..Default::default()
                },
                ShaderNode {
                    id: 2,
                    kind: ShaderNodeKind::Output,
                    pos: [700.0, 105.0],
                    ..Default::default()
                },
            ],
            links: vec![ShaderLink {
                from: 1,
                to: 2,
                from_socket: 0,
                to_socket: 0,
            }],
            next_id: 3,
        }
    }
}

/// Result of evaluating a graph into material slots (and optional new textures).
#[derive(Clone, Debug, Default)]
pub struct GraphEvalResult {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub base_color_tex: Option<usize>,
    pub metallic_roughness_tex: Option<usize>,
    pub normal_tex: Option<usize>,
    pub occlusion_tex: Option<usize>,
    pub emissive_tex: Option<usize>,
    /// Newly generated procedural textures to append to the scene.
    pub new_textures: Vec<TextureData>,
    pub displacement_scale: f32,
    pub volume_density: f32,
    /// Light Output node: emission color fed to its Surface socket.
    pub light_color: Option<[f32; 4]>,
    /// World Output node: background color fed to its Surface socket.
    pub world_color: Option<[f32; 4]>,
    /// World Output node: volume density.
    pub world_volume_density: f32,
    /// AOV Output nodes: (node id, color, value).
    pub aovs: Vec<(u64, [f32; 4], f32)>,
}

fn hash2(x: f32, y: f32) -> f32 {
    let mut n = x * 127.1 + y * 311.7;
    n = (n.sin() * 43758.5453).fract();
    n
}

fn noise2(x: f32, y: f32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let fx = x.fract();
    let fy = y.fract();
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let a = hash2(ix, iy);
    let b = hash2(ix + 1.0, iy);
    let c = hash2(ix, iy + 1.0);
    let d = hash2(ix + 1.0, iy + 1.0);
    a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy
}

fn fbm(x: f32, y: f32, octaves: i32) -> f32 {
    let mut v = 0.0;
    let mut a = 0.5;
    let mut f = 1.0;
    let mut x = x;
    let mut y = y;
    for _ in 0..octaves {
        v += a * noise2(x * f, y * f);
        a *= 0.5;
        f *= 2.0;
        x += 13.1;
        y += 7.7;
    }
    v.clamp(0.0, 1.0)
}

fn voronoi(x: f32, y: f32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let fx = x.fract();
    let fy = y.fract();
    let mut md = 1.0f32;
    for j in -1..=1 {
        for i in -1..=1 {
            let ox = hash2(ix + i as f32, iy + j as f32);
            let oy = hash2(ix + i as f32 + 19.2, iy + j as f32 + 7.4);
            let dx = (i as f32) + ox - fx;
            let dy = (j as f32) + oy - fy;
            md = md.min(dx * dx + dy * dy);
        }
    }
    md.sqrt().clamp(0.0, 1.0)
}

fn sample_tex(tex: &TextureData, u: f32, v: f32) -> [f32; 4] {
    if tex.width == 0 || tex.height == 0 || tex.rgba8.len() < 4 {
        return [1.0, 1.0, 1.0, 1.0];
    }
    let u = u.fract().rem_euclid(1.0);
    let v = v.fract().rem_euclid(1.0);
    let x = ((u * tex.width as f32) as u32).min(tex.width - 1);
    let y = ((v * tex.height as f32) as u32).min(tex.height - 1);
    let i = ((y * tex.width + x) * 4) as usize;
    if i + 3 >= tex.rgba8.len() {
        return [1.0, 1.0, 1.0, 1.0];
    }
    [
        tex.rgba8[i] as f32 / 255.0,
        tex.rgba8[i + 1] as f32 / 255.0,
        tex.rgba8[i + 2] as f32 / 255.0,
        tex.rgba8[i + 3] as f32 / 255.0,
    ]
}

fn hash01(x: f32, y: f32) -> f32 {
    hash2(x, y).rem_euclid(1.0)
}

fn sky_color(v: f32) -> [f32; 4] {
    let v = v.clamp(0.0, 1.0);
    let zenith = [0.15, 0.35, 0.80, 1.0];
    let horizon = [0.70, 0.80, 0.95, 1.0];
    let ground = [0.25, 0.22, 0.20, 1.0];
    if v < 0.5 {
        mix4(zenith, horizon, v / 0.5)
    } else {
        mix4(horizon, ground, (v - 0.5) / 0.5)
    }
}

/// Procedural / lookup color for the extra Texture-category nodes at UV (u, v).
fn texture_node_color(node: &ShaderNode, textures: &[TextureData], u: f32, v: f32) -> [f32; 4] {
    let scale = if node.value.abs() < 1e-4 { 5.0 } else { node.value };
    let g = |n: f32| [n, n, n, 1.0];
    match node.kind {
        ShaderNodeKind::Brick => {
            // 2:1 bricks, every other row offset by half a brick.
            let x = u * scale;
            let y = v * scale * 2.0;
            let row = y.floor();
            let offset = if (row as i64).rem_euclid(2) == 1 { 0.5 } else { 0.0 };
            let xo = x + offset;
            let col = xo.floor();
            let fx = xo - col;
            let fy = y - row;
            let mortar = 0.04;
            if fx < mortar || fy < mortar * 2.0 {
                [0.1, 0.1, 0.1, 1.0]
            } else {
                let t = 0.8 + 0.2 * hash01(col, row);
                [node.color[0] * t, node.color[1] * t, node.color[2] * t, 1.0]
            }
        }
        ShaderNodeKind::Checker => {
            let s = (u * scale).floor() as i64 + (v * scale).floor() as i64;
            if s.rem_euclid(2) == 0 {
                [node.color[0], node.color[1], node.color[2], 1.0]
            } else {
                [0.2, 0.2, 0.2, 1.0]
            }
        }
        ShaderNodeKind::Environment => {
            if let Some(ti) = node.image_index.filter(|i| *i < textures.len()) {
                sample_tex(&textures[ti], u, v)
            } else {
                sky_color(v)
            }
        }
        ShaderNodeKind::Sky => sky_color(v),
        ShaderNodeKind::Gabor => {
            let x = u * scale;
            let y = v * scale;
            let ix = x.floor();
            let iy = y.floor();
            let mut sum = 0.0f32;
            for j in -1..=1 {
                for i in -1..=1 {
                    let cx = ix + i as f32;
                    let cy = iy + j as f32;
                    let px = cx + hash01(cx, cy);
                    let py = cy + hash01(cx + 17.3, cy + 5.1);
                    let th = hash01(cx + 3.7, cy + 11.9) * std::f32::consts::PI;
                    let dx = x - px;
                    let dy = y - py;
                    let d2 = dx * dx + dy * dy;
                    let k = (-std::f32::consts::PI * 2.0 * d2).exp();
                    sum += k * (std::f32::consts::TAU * (dx * th.cos() + dy * th.sin())).cos();
                }
            }
            g((0.5 + 0.35 * sum).clamp(0.0, 1.0))
        }
        ShaderNodeKind::Gradient => g(u.clamp(0.0, 1.0)),
        ShaderNodeKind::Ies => {
            let dx = (u - 0.5) * 2.0;
            let dy = (v - 0.5) * 2.0;
            let r = (dx * dx + dy * dy).sqrt().min(1.0);
            g((r * std::f32::consts::FRAC_PI_2).cos().max(0.0).powf((scale * 0.2).max(0.1)))
        }
        ShaderNodeKind::Magic => {
            let px = u * scale;
            let py = v * scale;
            let pz = 0.5 * scale;
            let mut x = ((px + py + pz) * 5.0).sin();
            let mut y = ((-px + py - pz) * 5.0).cos();
            let mut z = -((-px - py + pz) * 5.0).cos();
            for _ in 0..2 {
                let nx = ((y - x + z) * 5.0).cos().abs();
                let ny = ((x - y - z) * 5.0).cos().abs();
                let nz = ((-x - y + z) * 5.0).cos().abs();
                x = nx;
                y = ny;
                z = nz;
            }
            [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0), z.clamp(0.0, 1.0), 1.0]
        }
        ShaderNodeKind::Wave => {
            let d = fbm(u * scale, v * scale, 3) * 2.0;
            g(0.5 + 0.5 * ((u * scale + d) * std::f32::consts::TAU).sin())
        }
        ShaderNodeKind::WhiteNoise => g(hash01((u * 512.0).floor(), (v * 512.0).floor())),
        _ => [1.0, 1.0, 1.0, 1.0],
    }
}

fn rgb_to_hsv(c: [f32; 3]) -> [f32; 3] {
    let max = c[0].max(c[1]).max(c[2]);
    let min = c[0].min(c[1]).min(c[2]);
    let d = max - min;
    let h = if d <= 1e-8 {
        0.0
    } else if max == c[0] {
        ((c[1] - c[2]) / d).rem_euclid(6.0) / 6.0
    } else if max == c[1] {
        ((c[2] - c[0]) / d + 2.0) / 6.0
    } else {
        ((c[0] - c[1]) / d + 4.0) / 6.0
    };
    let s = if max <= 1e-8 { 0.0 } else { d / max };
    [h, s, max]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h6 = h.rem_euclid(1.0) * 6.0;
    let i = h6.floor();
    let f = h6 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i as i32 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

fn rgb_to_hsl(c: [f32; 3]) -> [f32; 3] {
    let [h, s, v] = rgb_to_hsv(c);
    let l = v * (1.0 - s * 0.5);
    let sl = if l <= 1e-8 || l >= 1.0 - 1e-8 {
        0.0
    } else {
        (v - l) / l.min(1.0 - l)
    };
    [h, sl, l]
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    let v = l + s * l.min(1.0 - l);
    let sv = if v <= 1e-8 { 0.0 } else { 2.0 * (1.0 - l / v) };
    hsv_to_rgb(h, sv, v)
}

/// Approximate blackbody color (sRGB-ish, 0–1) for a temperature in Kelvin.
fn blackbody_rgb(kelvin: f32) -> [f32; 3] {
    let t = kelvin.clamp(800.0, 12000.0) / 100.0;
    let r = if t <= 66.0 { 255.0 } else { 329.698_73 * (t - 60.0).powf(-0.133_204_76) };
    let g = if t <= 66.0 {
        99.470_8 * t.ln() - 161.119_57
    } else {
        288.122_17 * (t - 60.0).powf(-0.075_514_85)
    };
    let b = if t >= 66.0 {
        255.0
    } else if t <= 19.0 {
        0.0
    } else {
        138.517_73 * (t - 10.0).ln() - 305.044_8
    };
    [
        (r / 255.0).clamp(0.0, 1.0),
        (g / 255.0).clamp(0.0, 1.0),
        (b / 255.0).clamp(0.0, 1.0),
    ]
}

/// Approximate visible-spectrum color for a wavelength in nanometres.
fn wavelength_rgb(nm: f32) -> [f32; 3] {
    let w = nm;
    let (r, g, b) = if (380.0..440.0).contains(&w) {
        ((440.0 - w) / 60.0, 0.0, 1.0)
    } else if (440.0..490.0).contains(&w) {
        (0.0, (w - 440.0) / 50.0, 1.0)
    } else if (490.0..510.0).contains(&w) {
        (0.0, 1.0, (510.0 - w) / 20.0)
    } else if (510.0..580.0).contains(&w) {
        ((w - 510.0) / 70.0, 1.0, 0.0)
    } else if (580.0..645.0).contains(&w) {
        (1.0, (645.0 - w) / 65.0, 0.0)
    } else if (645.0..=780.0).contains(&w) {
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 0.0, 0.0)
    };
    let f = if (380.0..420.0).contains(&w) {
        0.3 + 0.7 * (w - 380.0) / 40.0
    } else if (420.0..701.0).contains(&w) {
        1.0
    } else if (701.0..=780.0).contains(&w) {
        0.3 + 0.7 * (780.0 - w) / 80.0
    } else {
        0.0
    };
    [r * f, g * f, b * f]
}

fn curve_point(x: f32, mid: f32) -> f32 {
    (x + 4.0 * (mid - 0.5) * x * (1.0 - x)).clamp(0.0, 1.0)
}

/// Evaluate one of the extra Color-category nodes; returns (output socket, value) pairs.
fn eval_color_node(
    graph: &ShaderGraph,
    id: u64,
    node: &ShaderNode,
    cache: &HashMap<(u64, u8), SocketVal>,
) -> Vec<(u8, SocketVal)> {
    let in_color = |s: u8, d: [f32; 4]| {
        find_input(graph, id, s, cache).map(|v| v.as_color()).unwrap_or(d)
    };
    let in_scalar = |s: u8, d: f32| {
        find_input(graph, id, s, cache).map(|v| v.as_scalar()).unwrap_or(d)
    };
    let p = node.params;
    let mode = node.value.round() as i32;
    match node.kind {
        ShaderNodeKind::Blackbody => {
            let c = blackbody_rgb(in_scalar(0, node.value));
            vec![(0, SocketVal::Color([c[0], c[1], c[2], 1.0]))]
        }
        ShaderNodeKind::Wavelength => {
            let c = wavelength_rgb(in_scalar(0, node.value));
            vec![(0, SocketVal::Color([c[0], c[1], c[2], 1.0]))]
        }
        ShaderNodeKind::BrightContrast => {
            let c = in_color(0, node.color);
            let a = 1.0 + p[1];
            let b = p[0] - p[1] * 0.5;
            vec![(
                0,
                SocketVal::Color([
                    (a * c[0] + b).max(0.0),
                    (a * c[1] + b).max(0.0),
                    (a * c[2] + b).max(0.0),
                    c[3],
                ]),
            )]
        }
        ShaderNodeKind::Gamma => {
            let c = in_color(0, node.color);
            let g = node.value.max(0.01);
            vec![(
                0,
                SocketVal::Color([c[0].max(0.0).powf(g), c[1].max(0.0).powf(g), c[2].max(0.0).powf(g), c[3]]),
            )]
        }
        ShaderNodeKind::HueSatVal => {
            let c = in_color(0, node.color);
            let hsv = rgb_to_hsv([c[0], c[1], c[2]]);
            let h = (hsv[0] + p[0] - 0.5).rem_euclid(1.0);
            let rgb = hsv_to_rgb(h, (hsv[1] * p[1]).clamp(0.0, 1.0), hsv[2] * p[2]);
            let out = [rgb[0], rgb[1], rgb[2], c[3]];
            vec![(0, SocketVal::Color(mix4(c, out, p[3])))]
        }
        ShaderNodeKind::InvertColor => {
            let c = in_color(0, node.color);
            let inv = [1.0 - c[0], 1.0 - c[1], 1.0 - c[2], c[3]];
            vec![(0, SocketVal::Color(mix4(c, inv, node.value.clamp(0.0, 1.0))))]
        }
        ShaderNodeKind::LightFalloff => {
            let strength = in_scalar(0, node.value);
            // No light distance on the CPU: all three falloffs resolve to the strength.
            vec![
                (0, SocketVal::Scalar(strength)),
                (1, SocketVal::Scalar(strength)),
                (2, SocketVal::Scalar(strength)),
            ]
        }
        ShaderNodeKind::RgbCurves => {
            let c = in_color(0, node.color);
            let r = curve_point(curve_point(c[0].clamp(0.0, 1.0), p[0]), p[1]);
            let g = curve_point(curve_point(c[1].clamp(0.0, 1.0), p[0]), p[2]);
            let b = curve_point(curve_point(c[2].clamp(0.0, 1.0), p[0]), p[3]);
            vec![(0, SocketVal::Color([r, g, b, c[3]]))]
        }
        ShaderNodeKind::CombineColor => {
            let a = in_scalar(0, p[0]);
            let b = in_scalar(1, p[1]);
            let c = in_scalar(2, p[2]);
            let rgb = match mode {
                1 => hsv_to_rgb(a, b, c),
                2 => hsl_to_rgb(a, b, c),
                _ => [a, b, c],
            };
            vec![(0, SocketVal::Color([rgb[0], rgb[1], rgb[2], 1.0]))]
        }
        ShaderNodeKind::SeparateColor => {
            let c = in_color(0, node.color);
            let rgb = [c[0], c[1], c[2]];
            let v = match mode {
                1 => rgb_to_hsv(rgb),
                2 => rgb_to_hsl(rgb),
                _ => rgb,
            };
            vec![
                (0, SocketVal::Scalar(v[0])),
                (1, SocketVal::Scalar(v[1])),
                (2, SocketVal::Scalar(v[2])),
            ]
        }
        ShaderNodeKind::RgbToBw => {
            let c = in_color(0, node.color);
            vec![(0, SocketVal::Scalar(SocketVal::Color(c).as_scalar()))]
        }
        ShaderNodeKind::ShaderToRgb => {
            let c = in_color(0, [0.0, 0.0, 0.0, 1.0]);
            vec![
                (0, SocketVal::Color([c[0], c[1], c[2], 1.0])),
                (1, SocketVal::Scalar(c[3])),
            ]
        }
        _ => Vec::new(),
    }
}

pub const MATH_OPS: &[&str] = &[
    "Add", "Subtract", "Multiply", "Divide", "Power", "Logarithm", "Square Root", "Absolute",
    "Minimum", "Maximum", "Less Than", "Greater Than", "Sign", "Round", "Floor", "Ceil",
    "Truncate", "Fraction", "Modulo", "Sine", "Cosine", "Tangent", "Multiply Add",
];

pub const VECTOR_MATH_OPS: &[&str] = &[
    "Add", "Subtract", "Multiply", "Divide", "Cross Product", "Dot Product", "Distance", "Length",
    "Scale", "Normalize", "Absolute", "Minimum", "Maximum", "Floor", "Ceil", "Fraction", "Modulo",
    "Sine", "Cosine", "Tangent", "Reflect",
];

fn math_op(op: usize, a: f32, b: f32, c: f32) -> f32 {
    let r = match op {
        0 => a + b,
        1 => a - b,
        2 => a * b,
        3 => if b.abs() < 1e-12 { 0.0 } else { a / b },
        4 => a.powf(b),
        5 => if a > 0.0 && b > 0.0 && b != 1.0 { a.ln() / b.ln() } else { 0.0 },
        6 => a.max(0.0).sqrt(),
        7 => a.abs(),
        8 => a.min(b),
        9 => a.max(b),
        10 => if a < b { 1.0 } else { 0.0 },
        11 => if a > b { 1.0 } else { 0.0 },
        12 => if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { 0.0 },
        13 => a.round(),
        14 => a.floor(),
        15 => a.ceil(),
        16 => a.trunc(),
        17 => a - a.floor(),
        18 => if b.abs() < 1e-12 { 0.0 } else { a % b },
        19 => a.sin(),
        20 => a.cos(),
        21 => a.tan(),
        22 => a * b + c,
        _ => 0.0,
    };
    if r.is_finite() { r } else { 0.0 }
}

fn vlen(a: [f32; 3]) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn vdot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Normalize; returns the zero vector for a (near-)zero input.
fn vnorm(a: [f32; 3]) -> [f32; 3] {
    let l = vlen(a);
    if l < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        [a[0] / l, a[1] / l, a[2] / l]
    }
}

fn vmap(a: [f32; 3], f: impl Fn(f32) -> f32) -> [f32; 3] {
    [f(a[0]), f(a[1]), f(a[2])]
}

fn vmap2(a: [f32; 3], b: [f32; 3], f: impl Fn(f32, f32) -> f32) -> [f32; 3] {
    [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2])]
}

/// Vector Math operation; returns (vector result, scalar result).
fn vector_math(op: usize, a: [f32; 3], b: [f32; 3], scale: f32) -> ([f32; 3], f32) {
    let sv = |x: f32| ([x, x, x], x);
    match op {
        0 => (vmap2(a, b, |x, y| x + y), 0.0),
        1 => (vmap2(a, b, |x, y| x - y), 0.0),
        2 => (vmap2(a, b, |x, y| x * y), 0.0),
        3 => (vmap2(a, b, |x, y| if y.abs() < 1e-12 { 0.0 } else { x / y }), 0.0),
        4 => (
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ],
            0.0,
        ),
        5 => sv(vdot(a, b)),
        6 => sv(vlen(vmap2(a, b, |x, y| x - y))),
        7 => sv(vlen(a)),
        8 => (vmap(a, |x| x * scale), 0.0),
        9 => (vnorm(a), 0.0),
        10 => (vmap(a, f32::abs), 0.0),
        11 => (vmap2(a, b, f32::min), 0.0),
        12 => (vmap2(a, b, f32::max), 0.0),
        13 => (vmap(a, f32::floor), 0.0),
        14 => (vmap(a, f32::ceil), 0.0),
        15 => (vmap(a, |x| x - x.floor()), 0.0),
        16 => (vmap2(a, b, |x, y| if y.abs() < 1e-12 { 0.0 } else { x % y }), 0.0),
        17 => (vmap(a, f32::sin), 0.0),
        18 => (vmap(a, f32::cos), 0.0),
        19 => (vmap(a, f32::tan), 0.0),
        20 => {
            let n = vnorm(b);
            let d = 2.0 * vdot(a, n);
            ([a[0] - d * n[0], a[1] - d * n[1], a[2] - d * n[2]], 0.0)
        }
        _ => ([0.0; 3], 0.0),
    }
}

/// Tiny expression parser for the Script node: numbers, variables, + - * / % ^, parentheses
/// and a handful of math functions.
struct ExprParser<'a> {
    chars: Vec<char>,
    pos: usize,
    vars: &'a [(&'a str, f32)],
}

impl<'a> ExprParser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_ws();
        self.chars.get(self.pos).copied()
    }

    fn expr(&mut self) -> Option<f32> {
        let mut v = self.term()?;
        loop {
            match self.peek() {
                Some('+') => {
                    self.pos += 1;
                    v += self.term()?;
                }
                Some('-') => {
                    self.pos += 1;
                    v -= self.term()?;
                }
                _ => break,
            }
        }
        Some(v)
    }

    fn term(&mut self) -> Option<f32> {
        let mut v = self.factor()?;
        loop {
            match self.peek() {
                Some('*') => {
                    self.pos += 1;
                    v *= self.factor()?;
                }
                Some('/') => {
                    self.pos += 1;
                    let d = self.factor()?;
                    v = if d.abs() < 1e-12 { 0.0 } else { v / d };
                }
                Some('%') => {
                    self.pos += 1;
                    let d = self.factor()?;
                    v = if d.abs() < 1e-12 { 0.0 } else { v % d };
                }
                _ => break,
            }
        }
        Some(v)
    }

    fn factor(&mut self) -> Option<f32> {
        match self.peek() {
            Some('-') => {
                self.pos += 1;
                return Some(-self.factor()?);
            }
            Some('+') => {
                self.pos += 1;
                return self.factor();
            }
            _ => {}
        }
        let base = self.atom()?;
        if self.peek() == Some('^') {
            self.pos += 1;
            let e = self.factor()?;
            let r = base.powf(e);
            return Some(if r.is_finite() { r } else { 0.0 });
        }
        Some(base)
    }

    fn atom(&mut self) -> Option<f32> {
        let c = self.peek()?;
        if c == '(' {
            self.pos += 1;
            let v = self.expr()?;
            if self.peek()? != ')' {
                return None;
            }
            self.pos += 1;
            return Some(v);
        }
        if c.is_ascii_digit() || c == '.' {
            let start = self.pos;
            while self.pos < self.chars.len()
                && (self.chars[self.pos].is_ascii_digit() || self.chars[self.pos] == '.')
            {
                self.pos += 1;
            }
            let text: String = self.chars[start..self.pos].iter().collect();
            return text.parse::<f32>().ok();
        }
        if c.is_alphabetic() || c == '_' {
            let start = self.pos;
            while self.pos < self.chars.len()
                && (self.chars[self.pos].is_alphanumeric() || self.chars[self.pos] == '_')
            {
                self.pos += 1;
            }
            let name: String = self.chars[start..self.pos].iter().collect();
            if self.peek() == Some('(') {
                self.pos += 1;
                let mut args: Vec<f32> = Vec::new();
                if self.peek() != Some(')') {
                    loop {
                        args.push(self.expr()?);
                        match self.peek()? {
                            ',' => self.pos += 1,
                            ')' => break,
                            _ => return None,
                        }
                    }
                }
                self.pos += 1; // ')'
                let a = args.first().copied().unwrap_or(0.0);
                let b = args.get(1).copied().unwrap_or(0.0);
                let c3 = args.get(2).copied().unwrap_or(0.0);
                let r = match (name.as_str(), args.len()) {
                    ("sin", 1) => a.sin(),
                    ("cos", 1) => a.cos(),
                    ("tan", 1) => a.tan(),
                    ("abs", 1) => a.abs(),
                    ("sqrt", 1) => a.max(0.0).sqrt(),
                    ("floor", 1) => a.floor(),
                    ("ceil", 1) => a.ceil(),
                    ("fract", 1) => a - a.floor(),
                    ("exp", 1) => a.exp(),
                    ("ln", 1) => if a > 0.0 { a.ln() } else { 0.0 },
                    ("sign", 1) => if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { 0.0 },
                    ("min", 2) => a.min(b),
                    ("max", 2) => a.max(b),
                    ("pow", 2) => a.powf(b),
                    ("mod", 2) => if b.abs() < 1e-12 { 0.0 } else { a % b },
                    ("clamp", 3) => a.max(b).min(c3),
                    ("mix", 3) => a + (b - a) * c3,
                    _ => return None,
                };
                return Some(if r.is_finite() { r } else { 0.0 });
            }
            return match name.as_str() {
                "pi" => Some(std::f32::consts::PI),
                "tau" => Some(std::f32::consts::TAU),
                "e" => Some(std::f32::consts::E),
                _ => self.vars.iter().find(|(n, _)| *n == name).map(|(_, v)| *v),
            };
        }
        None
    }
}

/// Evaluate a Script-node expression. Returns `None` on a syntax error or unknown name.
pub fn eval_expr(src: &str, vars: &[(&str, f32)]) -> Option<f32> {
    let mut p = ExprParser { chars: src.chars().collect(), pos: 0, vars };
    let v = p.expr()?;
    p.skip_ws();
    if p.pos == p.chars.len() {
        Some(v)
    } else {
        None
    }
}

/// Evaluate one of the Vector / Utilities nodes; returns (output socket, value) pairs.
fn eval_vu_node(
    graph: &ShaderGraph,
    id: u64,
    node: &ShaderNode,
    cache: &HashMap<(u64, u8), SocketVal>,
    uv: [f32; 2],
) -> Vec<(u8, SocketVal)> {
    let in_vec = |s: u8, d: [f32; 3]| {
        find_input(graph, id, s, cache).map(|v| v.as_vector()).unwrap_or(d)
    };
    let in_scalar = |s: u8, d: f32| {
        find_input(graph, id, s, cache).map(|v| v.as_scalar()).unwrap_or(d)
    };
    let black = SocketVal::Color([0.0, 0.0, 0.0, 1.0]);
    let p = node.params;
    let cxyz = [node.color[0], node.color[1], node.color[2]];
    let op = node.value.round().max(0.0) as usize;
    match node.kind {
        ShaderNodeKind::CombineXyz => vec![(
            0,
            SocketVal::Vector([in_scalar(0, p[0]), in_scalar(1, p[1]), in_scalar(2, p[2])]),
        )],
        ShaderNodeKind::SeparateXyz => {
            let v = in_vec(0, cxyz);
            vec![
                (0, SocketVal::Scalar(v[0])),
                (1, SocketVal::Scalar(v[1])),
                (2, SocketVal::Scalar(v[2])),
            ]
        }
        ShaderNodeKind::VectorMath => {
            let a = in_vec(0, cxyz);
            let b = in_vec(1, [p[0], p[1], p[2]]);
            let (vec, val) = vector_math(op, a, b, in_scalar(2, p[3]));
            vec![(0, SocketVal::Vector(vec)), (1, SocketVal::Scalar(val))]
        }
        ShaderNodeKind::VectorRotate => {
            let v = in_vec(0, cxyz);
            let (sn, cs) = in_scalar(1, p[0]).sin_cos();
            let out = match op {
                0 => [v[0], v[1] * cs - v[2] * sn, v[1] * sn + v[2] * cs],
                1 => [v[0] * cs + v[2] * sn, v[1], -v[0] * sn + v[2] * cs],
                _ => [v[0] * cs - v[1] * sn, v[0] * sn + v[1] * cs, v[2]],
            };
            vec![(0, SocketVal::Vector(out))]
        }
        ShaderNodeKind::VectorTransform => {
            // World / Object / Camera matrices are not available on the CPU, so spaces are
            // treated as identical; Normal type still re-normalizes.
            let v = in_vec(0, cxyz);
            let out = if op == 2 { vnorm(v) } else { v };
            vec![(0, SocketVal::Vector(out))]
        }
        ShaderNodeKind::VectorCurves => {
            let v = in_vec(0, cxyz);
            let f = p[3].clamp(0.0, 1.0);
            let cv = |x: f32, m: f32| if (0.0..=1.0).contains(&x) { curve_point(x, m) } else { x };
            let c = [cv(v[0], p[0]), cv(v[1], p[1]), cv(v[2], p[2])];
            vec![(
                0,
                SocketVal::Vector([
                    v[0] + (c[0] - v[0]) * f,
                    v[1] + (c[1] - v[1]) * f,
                    v[2] + (c[2] - v[2]) * f,
                ]),
            )]
        }
        ShaderNodeKind::Normal => {
            let mut n = vnorm([p[0], p[1], p[2]]);
            if vlen(n) < 0.5 {
                n = [0.0, 0.0, 1.0];
            }
            let inp = vnorm(in_vec(0, [0.0, 0.0, 1.0]));
            vec![(0, SocketVal::Vector(n)), (1, SocketVal::Scalar(vdot(n, inp)))]
        }
        ShaderNodeKind::Displacement => {
            let h = in_scalar(0, p[0]);
            let mid = in_scalar(1, p[1]);
            let sc = in_scalar(2, p[2]);
            vec![(0, SocketVal::Scalar((h - mid) * sc))]
        }
        ShaderNodeKind::VectorDisplacement => {
            let v = in_vec(0, cxyz);
            let mid = in_scalar(1, p[0]);
            let sc = in_scalar(2, p[1]);
            vec![(0, SocketVal::Vector(vmap(v, |x| (x - mid) * sc)))]
        }
        ShaderNodeKind::Math => {
            let a = in_scalar(0, p[0]);
            let b = in_scalar(1, p[1]);
            let c = in_scalar(2, p[2]);
            vec![(0, SocketVal::Scalar(math_op(op, a, b, c)))]
        }
        ShaderNodeKind::MapRange => {
            let x = in_scalar(0, node.value);
            let fmin = in_scalar(1, p[0]);
            let fmax = in_scalar(2, p[1]);
            let tmin = in_scalar(3, p[2]);
            let tmax = in_scalar(4, p[3]);
            let t = if (fmax - fmin).abs() < 1e-8 {
                0.0
            } else {
                ((x - fmin) / (fmax - fmin)).clamp(0.0, 1.0)
            };
            vec![(0, SocketVal::Scalar(tmin + t * (tmax - tmin)))]
        }
        ShaderNodeKind::Clamp => {
            let x = in_scalar(0, node.value);
            let lo = in_scalar(1, p[0]);
            let hi = in_scalar(2, p[1]);
            vec![(0, SocketVal::Scalar(x.max(lo).min(hi)))]
        }
        ShaderNodeKind::FloatCurve => {
            let fac = in_scalar(0, p[0]).clamp(0.0, 1.0);
            let x = in_scalar(1, node.value);
            let c = curve_point(x.clamp(0.0, 1.0), p[1]);
            vec![(0, SocketVal::Scalar(x + (c - x) * fac))]
        }
        ShaderNodeKind::Switch => {
            let on = in_scalar(0, p[0]) >= 0.5;
            let f = find_input(graph, id, 1, cache).unwrap_or(black);
            let t = find_input(graph, id, 2, cache)
                .unwrap_or(SocketVal::Color([1.0, 1.0, 1.0, 1.0]));
            vec![(0, if on { t } else { f })]
        }
        ShaderNodeKind::IndexSwitch => {
            let idx = in_scalar(0, p[0]).round().clamp(0.0, 3.0) as u8;
            vec![(0, find_input(graph, id, 1 + idx, cache).unwrap_or(black))]
        }
        ShaderNodeKind::MenuSwitch => {
            let idx = p[0].round().clamp(0.0, 3.0) as u8;
            vec![(0, find_input(graph, id, idx, cache).unwrap_or(black))]
        }
        ShaderNodeKind::Script => {
            let vars = [
                ("a", in_scalar(0, p[0])),
                ("b", in_scalar(1, p[1])),
                ("c", in_scalar(2, p[2])),
                ("u", uv[0]),
                ("v", uv[1]),
            ];
            let val = eval_expr(&node.text, &vars).unwrap_or(0.0);
            vec![
                (0, SocketVal::Scalar(val)),
                (1, SocketVal::Color([val, val, val, 1.0])),
            ]
        }
        ShaderNodeKind::Group => {
            // Simplified group: output N passes input N straight through.
            (0u8..3)
                .map(|i| (i, find_input(graph, id, i, cache).unwrap_or(SocketVal::Scalar(0.0))))
                .collect()
        }
        _ => Vec::new(),
    }
}

fn color_ramp(t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    // Simple black→white ramp (extensible later).
    [t, t, t, 1.0]
}

fn mix4(a: [f32; 4], b: [f32; 4], f: f32) -> [f32; 4] {
    let f = f.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
        a[3] + (b[3] - a[3]) * f,
    ]
}

#[derive(Clone, Copy)]
enum SocketVal {
    Scalar(f32),
    Color([f32; 4]),
    Vector([f32; 3]),
}

impl SocketVal {
    fn as_scalar(self) -> f32 {
        match self {
            Self::Scalar(s) => s,
            Self::Color(c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2],
            Self::Vector(v) => v[0],
        }
    }
    fn as_color(self) -> [f32; 4] {
        match self {
            Self::Scalar(s) => [s, s, s, 1.0],
            Self::Color(c) => c,
            Self::Vector(v) => [v[0], v[1], v[2], 1.0],
        }
    }
    fn as_vector(self) -> [f32; 3] {
        match self {
            Self::Scalar(s) => [s, s, s],
            Self::Color(c) => [c[0], c[1], c[2]],
            Self::Vector(v) => v,
        }
    }
}

fn find_input(
    graph: &ShaderGraph,
    node_id: u64,
    socket: u8,
    cache: &HashMap<(u64, u8), SocketVal>,
) -> Option<SocketVal> {
    for link in &graph.links {
        if link.to == node_id && link.to_socket == socket {
            return cache.get(&(link.from, link.from_socket)).copied();
        }
    }
    None
}

fn topo_order(graph: &ShaderGraph) -> Vec<u64> {
    let mut indeg: HashMap<u64, usize> = graph.nodes.iter().map(|n| (n.id, 0)).collect();
    let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
    for l in &graph.links {
        *indeg.entry(l.to).or_insert(0) += 1;
        adj.entry(l.from).or_default().push(l.to);
    }
    let mut q: Vec<u64> = indeg
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(id, _)| *id)
        .collect();
    let mut out = Vec::new();
    while let Some(id) = q.pop() {
        out.push(id);
        if let Some(nexts) = adj.get(&id) {
            for &n in nexts {
                if let Some(d) = indeg.get_mut(&n) {
                    *d = d.saturating_sub(1);
                    if *d == 0 {
                        q.push(n);
                    }
                }
            }
        }
    }
    // Append any remaining (cycles) so evaluation still runs something.
    for n in &graph.nodes {
        if !out.contains(&n.id) {
            out.push(n.id);
        }
    }
    out
}

/// Evaluate the graph against existing scene textures. Procedural results that need
/// a texture are generated at `tex_size`² and returned in `new_textures`.
pub fn evaluate_graph(
    graph: &ShaderGraph,
    material: &Material,
    textures: &[TextureData],
    tex_size: u32,
) -> GraphEvalResult {
    let mut result = GraphEvalResult {
        base_color: material.base_color,
        metallic: material.metallic,
        roughness: material.roughness,
        emissive: material.emissive,
        normal_scale: material.normal_scale,
        occlusion_strength: material.occlusion_strength,
        base_color_tex: material.base_color_tex,
        metallic_roughness_tex: material.metallic_roughness_tex,
        normal_tex: material.normal_tex,
        occlusion_tex: material.occlusion_tex,
        emissive_tex: material.emissive_tex,
        new_textures: Vec::new(),
        displacement_scale: 0.0,
        volume_density: 0.0,
        light_color: None,
        world_color: None,
        world_volume_density: 0.0,
        aovs: Vec::new(),
    };

    let order = topo_order(graph);
    let node_map: HashMap<u64, &ShaderNode> = graph.nodes.iter().map(|n| (n.id, n)).collect();
    let mut cache: HashMap<(u64, u8), SocketVal> = HashMap::new();
    // For procedural texture nodes we may bake a full image once.
    let mut proc_tex_index: HashMap<u64, usize> = HashMap::new();

    for id in order {
        let Some(node) = node_map.get(&id) else { continue };
        match node.kind {
            ShaderNodeKind::Value => {
                cache.insert((id, 0), SocketVal::Scalar(node.value));
            }
            ShaderNodeKind::Rgb => {
                cache.insert((id, 0), SocketVal::Color(node.color));
            }
            ShaderNodeKind::TexCoord => {
                // UV is supplied per-sample when baking; as a constant we expose (0.5,0.5,0).
                cache.insert((id, 0), SocketVal::Vector([0.5, 0.5, 0.0]));
            }
            ShaderNodeKind::Mapping => {
                let inp = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Vector([0.5, 0.5, 0.0]))
                    .as_vector();
                let s = if node.value.abs() < 1e-6 { 1.0 } else { node.value };
                cache.insert((id, 0), SocketVal::Vector([inp[0] * s, inp[1] * s, inp[2] * s]));
            }
            ShaderNodeKind::ImageTexture => {
                if let Some(ti) = node.image_index.filter(|i| *i < textures.len()) {
                    // Expose average color as scalar/color fallback; real UV sampling happens in bake.
                    let tex = &textures[ti];
                    let c = sample_tex(tex, 0.5, 0.5);
                    cache.insert((id, 0), SocketVal::Color(c));
                    // Remember index for material slot wiring.
                    proc_tex_index.insert(id, ti);
                } else {
                    cache.insert((id, 0), SocketVal::Color([1.0, 1.0, 1.0, 1.0]));
                }
            }
            ShaderNodeKind::Noise | ShaderNodeKind::Voronoi => {
                // Bake a procedural texture once.
                let size = tex_size.max(64).min(1024);
                let scale = if node.value.abs() < 1e-4 { 5.0 } else { node.value };
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32;
                        let v = y as f32 / size as f32;
                        let n = if node.kind == ShaderNodeKind::Noise {
                            fbm(u * scale, v * scale, 5)
                        } else {
                            voronoi(u * scale, v * scale)
                        };
                        let i = ((y * size + x) * 4) as usize;
                        let b = (n * 255.0) as u8;
                        rgba[i] = b;
                        rgba[i + 1] = b;
                        rgba[i + 2] = b;
                        rgba[i + 3] = 255;
                    }
                }
                let name = if node.kind == ShaderNodeKind::Noise {
                    format!("Noise_{id}")
                } else {
                    format!("Voronoi_{id}")
                };
                let tex = TextureData {
                    name,
                    width: size,
                    height: size,
                    rgba8: rgba,
                };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Color(mid));
            }
            ShaderNodeKind::Brick | ShaderNodeKind::Checker | ShaderNodeKind::Environment | ShaderNodeKind::Gabor | ShaderNodeKind::Gradient | ShaderNodeKind::Ies | ShaderNodeKind::Magic | ShaderNodeKind::Sky | ShaderNodeKind::Wave | ShaderNodeKind::WhiteNoise => {
                let k = node.kind;
                // Bake a procedural texture once (same path as Noise / Voronoi).
                let size = tex_size.max(64).min(1024);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32;
                        let v = y as f32 / size as f32;
                        let c = texture_node_color(node, textures, u, v);
                        let i = ((y * size + x) * 4) as usize;
                        rgba[i] = (c[0].clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 1] = (c[1].clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 2] = (c[2].clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 3] = 255;
                    }
                }
                let tex = TextureData {
                    name: format!("{}_{id}", k.title()),
                    width: size,
                    height: size,
                    rgba8: rgba,
                };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Color(mid));
                cache.insert((id, 1), SocketVal::Scalar(SocketVal::Color(mid).as_scalar()));
            }
            ShaderNodeKind::CombineXyz | ShaderNodeKind::SeparateXyz | ShaderNodeKind::VectorMath | ShaderNodeKind::VectorRotate | ShaderNodeKind::VectorTransform | ShaderNodeKind::VectorCurves | ShaderNodeKind::Normal | ShaderNodeKind::Displacement | ShaderNodeKind::VectorDisplacement | ShaderNodeKind::Math | ShaderNodeKind::MapRange | ShaderNodeKind::Clamp | ShaderNodeKind::FloatCurve | ShaderNodeKind::Switch | ShaderNodeKind::IndexSwitch | ShaderNodeKind::MenuSwitch | ShaderNodeKind::Script | ShaderNodeKind::Group => {
                for (i, v) in eval_vu_node(graph, id, node, &cache, [0.5, 0.5]) {
                    cache.insert((id, i), v);
                }
            }
            ShaderNodeKind::Blackbody | ShaderNodeKind::BrightContrast | ShaderNodeKind::Gamma | ShaderNodeKind::HueSatVal | ShaderNodeKind::InvertColor | ShaderNodeKind::LightFalloff | ShaderNodeKind::RgbCurves | ShaderNodeKind::Wavelength | ShaderNodeKind::CombineColor | ShaderNodeKind::SeparateColor | ShaderNodeKind::RgbToBw | ShaderNodeKind::ShaderToRgb => {
                for (i, v) in eval_color_node(graph, id, node, &cache) {
                    cache.insert((id, i), v);
                }
            }
            ShaderNodeKind::ColorRamp => {
                let fac = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Scalar(0.5))
                    .as_scalar();
                cache.insert((id, 0), SocketVal::Color(color_ramp(fac)));
            }
            ShaderNodeKind::Mix => {
                let fac = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Scalar(0.5))
                    .as_scalar();
                let a = find_input(graph, id, 1, &cache)
                    .unwrap_or(SocketVal::Color([0.0, 0.0, 0.0, 1.0]))
                    .as_color();
                let b = find_input(graph, id, 2, &cache)
                    .unwrap_or(SocketVal::Color([1.0, 1.0, 1.0, 1.0]))
                    .as_color();
                cache.insert((id, 0), SocketVal::Color(mix4(a, b, fac)));
            }
            ShaderNodeKind::Bump => {
                let h = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Scalar(0.5))
                    .as_scalar();
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 1.0]));
                result.normal_scale = (h * node.value).clamp(0.0, 2.0);
            }
            ShaderNodeKind::NormalMap => {
                if let Some(c) = find_input(graph, id, 0, &cache) {
                    let _ = c;
                }
                result.normal_scale = node.value.clamp(0.0, 2.0);
                // If upstream image, use as normal tex.
                for link in &graph.links {
                    if link.to == id {
                        if let Some(&ti) = proc_tex_index.get(&link.from) {
                            result.normal_tex = Some(ti);
                        }
                    }
                }
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 1.0]));
            }
            ShaderNodeKind::Emission => {
                let col = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Color(node.color))
                    .as_color();
                let s = node.value.max(0.0);
                cache.insert((id, 0), SocketVal::Color([col[0] * s, col[1] * s, col[2] * s, 1.0]));
                result.emissive = [col[0] * s, col[1] * s, col[2] * s];
            }
            ShaderNodeKind::Diffuse => {
                let col = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Color(node.color))
                    .as_color();
                cache.insert((id, 0), SocketVal::Color(col));
                result.base_color = col;
                result.metallic = 0.0;
                result.roughness = 0.9;
            }
            ShaderNodeKind::Glossy => {
                let col = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Color(node.color))
                    .as_color();
                cache.insert((id, 0), SocketVal::Color(col));
                result.base_color = col;
                result.metallic = 1.0;
                result.roughness = node.value.clamp(0.045, 1.0);
            }
            ShaderNodeKind::Principled => {
                if let Some(c) = find_input(graph, id, 0, &cache) {
                    result.base_color = c.as_color();
                }
                // Wire image textures into base_color_tex when linked to base color.
                for link in &graph.links {
                    if link.to == id {
                        if let Some(&ti) = proc_tex_index.get(&link.from) {
                            if let Some(src) = node_map.get(&link.from) {
                                match src.kind {
                                    ShaderNodeKind::ImageTexture
                                    | ShaderNodeKind::Noise
                                    | ShaderNodeKind::Voronoi
                                    | ShaderNodeKind::Mix
                                    | ShaderNodeKind::ColorRamp => {
                                        result.base_color_tex = Some(ti);
                                    }
                                    ShaderNodeKind::NormalMap => {
                                        result.normal_tex = Some(ti);
                                    }
                                    k if k.is_texture_extra() => {
                                        result.base_color_tex = Some(ti);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                if let Some(m) = find_input(graph, id, 1, &cache) {
                    result.metallic = m.as_scalar().clamp(0.0, 1.0);
                }
                if let Some(r) = find_input(graph, id, 2, &cache) {
                    result.roughness = r.as_scalar().clamp(0.045, 1.0);
                }
                if let Some(e) = find_input(graph, id, 3, &cache) {
                    let c = e.as_color();
                    result.emissive = [c[0], c[1], c[2]];
                }
                cache.insert((id, 0), SocketVal::Color(result.base_color));
            }
                        // Input nodes: sensible CPU defaults for bake / graph apply (not full path tracing).
            ShaderNodeKind::AmbientOcclusion => {
                cache.insert((id, 0), SocketVal::Color([1.0, 1.0, 1.0, 1.0]));
            }
            ShaderNodeKind::Attribute | ShaderNodeKind::ColorAttribute => {
                cache.insert((id, 0), SocketVal::Color(node.color));
            }
            ShaderNodeKind::Bevel | ShaderNodeKind::Tangent => {
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 1.0]));
            }
            ShaderNodeKind::CameraData => {
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 1.0])); // View Vector
                cache.insert((id, 1), SocketVal::Scalar(1.0)); // View Z Depth
                cache.insert((id, 2), SocketVal::Scalar(1.0)); // View Distance
            }
            ShaderNodeKind::Fresnel | ShaderNodeKind::LayerWeight => {
                let ior = if node.value.abs() < 1e-6 { 1.45 } else { node.value };
                // Schlick approx at facing angle ~0.5
                let f0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
                let f = f0 + (1.0 - f0) * 0.5;
                cache.insert((id, 0), SocketVal::Scalar(f.clamp(0.0, 1.0)));
                cache.insert((id, 1), SocketVal::Scalar((1.0 - f).clamp(0.0, 1.0))); // Facing for Layer Weight
            }
            ShaderNodeKind::Geometry => {
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 0.0])); // Position
                cache.insert((id, 1), SocketVal::Vector([0.0, 0.0, 1.0])); // Normal
                cache.insert((id, 2), SocketVal::Vector([1.0, 0.0, 0.0])); // Tangent
                cache.insert((id, 3), SocketVal::Scalar(0.0)); // Backfacing
            }
            ShaderNodeKind::CurvesInfo => {
                cache.insert((id, 0), SocketVal::Scalar(0.0)); // Is Strand
                cache.insert((id, 1), SocketVal::Scalar(0.5)); // Intercept
                cache.insert((id, 2), SocketVal::Scalar(1.0)); // Length
            }
            ShaderNodeKind::LightEvaluation | ShaderNodeKind::LightInfo => {
                cache.insert((id, 0), SocketVal::Color([1.0, 1.0, 1.0, 1.0]));
                cache.insert((id, 1), SocketVal::Scalar(1.0));
            }
            ShaderNodeKind::LightPath => {
                // Camera ray = 1 for surface shading approximation
                cache.insert((id, 0), SocketVal::Scalar(1.0)); // Is Camera Ray
                cache.insert((id, 1), SocketVal::Scalar(0.0)); // Is Shadow Ray
                cache.insert((id, 2), SocketVal::Scalar(0.0)); // Is Diffuse Ray
            }
            ShaderNodeKind::ObjectInfo => {
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 0.0])); // Location
                cache.insert((id, 1), SocketVal::Color(node.color)); // Color
                cache.insert((id, 2), SocketVal::Scalar(0.0)); // Object Index
                cache.insert((id, 3), SocketVal::Scalar(node.value)); // Random
            }
            ShaderNodeKind::ParticleInfo => {
                cache.insert((id, 0), SocketVal::Scalar(0.0)); // Index
                cache.insert((id, 1), SocketVal::Scalar(0.0)); // Age
                cache.insert((id, 2), SocketVal::Scalar(1.0)); // Lifetime
            }
            ShaderNodeKind::PointInfo => {
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 0.0]));
                cache.insert((id, 1), SocketVal::Scalar(1.0)); // Radius
                cache.insert((id, 2), SocketVal::Scalar(node.value)); // Random
            }
            ShaderNodeKind::Raycast | ShaderNodeKind::ShadowRaycast => {
                cache.insert((id, 0), SocketVal::Scalar(0.0)); // Hit
                cache.insert((id, 1), SocketVal::Vector([0.0, 0.0, 0.0])); // Position
                cache.insert((id, 2), SocketVal::Vector([0.0, 0.0, 1.0])); // Normal
                cache.insert((id, 3), SocketVal::Scalar(0.0)); // Distance
            }
            ShaderNodeKind::SceneTime => {
                cache.insert((id, 0), SocketVal::Scalar(0.0)); // Seconds
                cache.insert((id, 1), SocketVal::Scalar(1.0)); // Frame
            }
            ShaderNodeKind::UvMap => {
                cache.insert((id, 0), SocketVal::Vector([0.5, 0.5, 0.0]));
            }
            ShaderNodeKind::VolumeInfo => {
                cache.insert((id, 0), SocketVal::Color(node.color));
                cache.insert((id, 1), SocketVal::Scalar(node.value)); // Density
            }
            ShaderNodeKind::Wireframe => {
                cache.insert((id, 0), SocketVal::Scalar(0.0));
            }
            ShaderNodeKind::AddShader => {
                let a = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Color([0.0, 0.0, 0.0, 1.0]))
                    .as_color();
                let b = find_input(graph, id, 1, &cache)
                    .unwrap_or(SocketVal::Color([0.0, 0.0, 0.0, 1.0]))
                    .as_color();
                cache.insert(
                    (id, 0),
                    SocketVal::Color([
                        (a[0] + b[0]).min(1.0),
                        (a[1] + b[1]).min(1.0),
                        (a[2] + b[2]).min(1.0),
                        a[3].max(b[3]),
                    ]),
                );
            }
            ShaderNodeKind::MixShader => {
                let fac = find_input(graph, id, 0, &cache)
                    .map(|v| v.as_scalar())
                    .unwrap_or(node.value);
                let a = find_input(graph, id, 1, &cache)
                    .unwrap_or(SocketVal::Color([0.0, 0.0, 0.0, 1.0]))
                    .as_color();
                let b = find_input(graph, id, 2, &cache)
                    .unwrap_or(SocketVal::Color([0.0, 0.0, 0.0, 1.0]))
                    .as_color();
                cache.insert((id, 0), SocketVal::Color(mix4(a, b, fac)));
            }
            ShaderNodeKind::Background => {
                let col = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Color(node.color))
                    .as_color();
                let s = node.value.max(0.0);
                cache.insert((id, 0), SocketVal::Color([col[0] * s, col[1] * s, col[2] * s, 1.0]));
            }
            ShaderNodeKind::LightAccumulation => {
                // No CPU equivalent: contributes no light (black).
                cache.insert((id, 0), SocketVal::Color([0.0, 0.0, 0.0, 1.0]));
            }
            ShaderNodeKind::RayPortalBsdf => {
                // Ray portals pass rays through: behaves as fully transparent.
                cache.insert((id, 0), SocketVal::Color([0.0, 0.0, 0.0, 0.0]));
            }
            ShaderNodeKind::GlassBsdf
            | ShaderNodeKind::RefractionBsdf
            | ShaderNodeKind::HairBsdf
            | ShaderNodeKind::PrincipledHair
            | ShaderNodeKind::Holdout
            | ShaderNodeKind::MetallicBsdf
            | ShaderNodeKind::SpecularBsdf
            | ShaderNodeKind::SubsurfaceScattering
            | ShaderNodeKind::ToonBsdf
            | ShaderNodeKind::TranslucentBsdf
            | ShaderNodeKind::TransparentBsdf
            | ShaderNodeKind::SheenBsdf => {
                let mut col = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Color(node.color))
                    .as_color();
                let rough = node.value.clamp(0.045, 1.0);
                // (metallic, roughness, alpha) — simplified mapping onto PBR slots.
                let (metallic, roughness, alpha) = match node.kind {
                    ShaderNodeKind::GlassBsdf => (0.0, rough.min(0.3), 0.25),
                    ShaderNodeKind::RefractionBsdf => (0.0, rough.min(0.3), 0.35),
                    ShaderNodeKind::HairBsdf | ShaderNodeKind::PrincipledHair => (0.0, rough, 1.0),
                    ShaderNodeKind::MetallicBsdf => (1.0, rough, 1.0),
                    ShaderNodeKind::SpecularBsdf => (0.0, rough, 1.0),
                    ShaderNodeKind::SubsurfaceScattering => (0.0, 0.5, 1.0),
                    ShaderNodeKind::ToonBsdf => (0.0, 1.0, 1.0),
                    ShaderNodeKind::TranslucentBsdf => (0.0, 0.9, 1.0),
                    ShaderNodeKind::SheenBsdf => (0.0, 0.8, 1.0),
                    ShaderNodeKind::TransparentBsdf => (0.0, 0.5, 0.0),
                    _ => (0.0, 1.0, 1.0), // Holdout
                };
                if node.kind == ShaderNodeKind::Holdout {
                    col = [0.0, 0.0, 0.0, 0.0];
                } else {
                    col[3] = alpha;
                }
                cache.insert((id, 0), SocketVal::Color(col));
                result.base_color = col;
                result.metallic = metallic;
                result.roughness = roughness;
            }
            ShaderNodeKind::PrincipledVolume
            | ShaderNodeKind::VolumeAbsorption
            | ShaderNodeKind::VolumeScatter => {
                cache.insert((id, 0), SocketVal::Scalar(node.value.max(0.0)));
            }
            ShaderNodeKind::VolumeCoefficients => {
                let a = find_input(graph, id, 0, &cache).map(|v| v.as_scalar()).unwrap_or(0.0);
                let b = find_input(graph, id, 1, &cache).map(|v| v.as_scalar()).unwrap_or(0.0);
                cache.insert((id, 0), SocketVal::Scalar((node.value + a + b).max(0.0)));
            }
            ShaderNodeKind::LightOutput => {
                // Surface (socket 0)
                if let Some(s) = find_input(graph, id, 0, &cache) {
                    result.light_color = Some(s.as_color());
                }
            }
            ShaderNodeKind::WorldOutput => {
                // Surface (socket 0), Volume (socket 1)
                if let Some(s) = find_input(graph, id, 0, &cache) {
                    result.world_color = Some(s.as_color());
                }
                if let Some(v) = find_input(graph, id, 1, &cache) {
                    result.world_volume_density = v.as_scalar();
                }
            }
            ShaderNodeKind::AovOutput => {
                // Color (socket 0), Value (socket 1)
                let c = find_input(graph, id, 0, &cache)
                    .map(|v| v.as_color())
                    .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                let val = find_input(graph, id, 1, &cache)
                    .map(|v| v.as_scalar())
                    .unwrap_or(0.0);
                result.aovs.push((id, c, val));
            }
            ShaderNodeKind::Output => {
                // Surface (socket 0)
                if let Some(s) = find_input(graph, id, 0, &cache) {
                    let c = s.as_color();
                    // If nothing else set base from surface.
                    if result.base_color_tex.is_none() {
                        result.base_color = c;
                    }
                }
                // Volume (socket 1) → density scalar for future use
                if let Some(v) = find_input(graph, id, 1, &cache) {
                    result.volume_density = v.as_scalar();
                }
                // Displacement (socket 2)
                if let Some(d) = find_input(graph, id, 2, &cache) {
                    result.displacement_scale = match d {
                        SocketVal::Vector(v) => vlen(v),
                        other => other.as_scalar(),
                    };
                }
            }
        }
    }

    result
}

/// Apply evaluation result onto a material and append any new textures to `textures`.
pub fn apply_eval_to_material(
    material: &mut Material,
    textures: &mut Vec<TextureData>,
    eval: GraphEvalResult,
) {
    material.base_color = eval.base_color;
    if eval.base_color[3] < 0.999 {
        material.alpha_mode = crate::material::AlphaMode::Blend;
    }
    material.metallic = eval.metallic;
    material.roughness = eval.roughness;
    material.emissive = eval.emissive;
    material.normal_scale = eval.normal_scale;
    material.occlusion_strength = eval.occlusion_strength;

    let base = textures.len();
    for t in eval.new_textures {
        textures.push(t);
    }
    let map_idx = |idx: Option<usize>| -> Option<usize> {
        idx.map(|i| if i >= base { i } else { i })
    };
    // Indices in eval already account for append order (textures.len() at creation time).
    if let Some(i) = eval.base_color_tex {
        material.base_color_tex = Some(i);
    }
    if let Some(i) = eval.metallic_roughness_tex {
        material.metallic_roughness_tex = Some(i);
    }
    if let Some(i) = eval.normal_tex {
        material.normal_tex = Some(i);
    }
    if let Some(i) = eval.occlusion_tex {
        material.occlusion_tex = Some(i);
    }
    if let Some(i) = eval.emissive_tex {
        material.emissive_tex = Some(i);
    }
    let _ = (map_idx, base);
}

/// Bake the graph's color output into a new RGBA texture of `size`×`size`.
pub fn bake_graph_texture(
    graph: &ShaderGraph,
    material: &Material,
    textures: &[TextureData],
    size: u32,
) -> TextureData {
    let size = size.clamp(64, 2048);
    let mut rgba = vec![0u8; (size * size * 4) as usize];
    let node_map: HashMap<u64, &ShaderNode> = graph.nodes.iter().map(|n| (n.id, n)).collect();
    // Pre-evaluate non-UV-dependent parts once.
    let base_eval = evaluate_graph(graph, material, textures, size);

    for y in 0..size {
        for x in 0..size {
            let u = (x as f32 + 0.5) / size as f32;
            let v = (y as f32 + 0.5) / size as f32;
            let color = sample_graph_at(graph, &node_map, textures, &base_eval, u, v);
            let i = ((y * size + x) * 4) as usize;
            rgba[i] = (color[0].clamp(0.0, 1.0) * 255.0) as u8;
            rgba[i + 1] = (color[1].clamp(0.0, 1.0) * 255.0) as u8;
            rgba[i + 2] = (color[2].clamp(0.0, 1.0) * 255.0) as u8;
            rgba[i + 3] = (color[3].clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    TextureData {
        name: format!("Bake_{size}"),
        width: size,
        height: size,
        rgba8: rgba,
    }
}

fn sample_graph_at(
    graph: &ShaderGraph,
    node_map: &HashMap<u64, &ShaderNode>,
    textures: &[TextureData],
    base: &GraphEvalResult,
    u: f32,
    v: f32,
) -> [f32; 4] {
    // Walk from Image/Noise nodes through Mix/ColorRamp toward Principled/Output.
    // Lightweight per-pixel path for common texture chains.
    let order = topo_order(graph);
    let mut cache: HashMap<(u64, u8), SocketVal> = HashMap::new();
    for id in order {
        let Some(node) = node_map.get(&id) else { continue };
        match node.kind {
            ShaderNodeKind::TexCoord => {
                cache.insert((id, 0), SocketVal::Vector([u, v, 0.0]));
            }
            ShaderNodeKind::Mapping => {
                let inp = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Vector([u, v, 0.0]))
                    .as_vector();
                let s = if node.value.abs() < 1e-6 { 1.0 } else { node.value };
                cache.insert((id, 0), SocketVal::Vector([inp[0] * s, inp[1] * s, inp[2] * s]));
            }
            ShaderNodeKind::ImageTexture => {
                let uv = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Vector([u, v, 0.0]))
                    .as_vector();
                let c = if let Some(ti) = node.image_index.filter(|i| *i < textures.len()) {
                    sample_tex(&textures[ti], uv[0], uv[1])
                } else {
                    [1.0, 1.0, 1.0, 1.0]
                };
                cache.insert((id, 0), SocketVal::Color(c));
            }
            ShaderNodeKind::Noise => {
                let uv = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Vector([u, v, 0.0]))
                    .as_vector();
                let scale = if node.value.abs() < 1e-4 { 5.0 } else { node.value };
                let n = fbm(uv[0] * scale, uv[1] * scale, 5);
                cache.insert((id, 0), SocketVal::Color([n, n, n, 1.0]));
            }
            ShaderNodeKind::Voronoi => {
                let uv = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Vector([u, v, 0.0]))
                    .as_vector();
                let scale = if node.value.abs() < 1e-4 { 5.0 } else { node.value };
                let n = voronoi(uv[0] * scale, uv[1] * scale);
                cache.insert((id, 0), SocketVal::Color([n, n, n, 1.0]));
            }
            ShaderNodeKind::Value => {
                cache.insert((id, 0), SocketVal::Scalar(node.value));
            }
            ShaderNodeKind::Rgb => {
                cache.insert((id, 0), SocketVal::Color(node.color));
            }
            ShaderNodeKind::ColorRamp => {
                let fac = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Scalar(0.5))
                    .as_scalar();
                cache.insert((id, 0), SocketVal::Color(color_ramp(fac)));
            }
            ShaderNodeKind::Mix => {
                let fac = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Scalar(0.5))
                    .as_scalar();
                let a = find_input(graph, id, 1, &cache)
                    .unwrap_or(SocketVal::Color([0.0, 0.0, 0.0, 1.0]))
                    .as_color();
                let b = find_input(graph, id, 2, &cache)
                    .unwrap_or(SocketVal::Color([1.0, 1.0, 1.0, 1.0]))
                    .as_color();
                cache.insert((id, 0), SocketVal::Color(mix4(a, b, fac)));
            }
            ShaderNodeKind::Principled => {
                let c = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Color(base.base_color))
                    .as_color();
                cache.insert((id, 0), SocketVal::Color(c));
            }
            ShaderNodeKind::Diffuse | ShaderNodeKind::Glossy | ShaderNodeKind::Emission => {
                let c = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Color(node.color))
                    .as_color();
                cache.insert((id, 0), SocketVal::Color(c));
            }
            k if k.is_vector_util() => {
                for (i, val) in eval_vu_node(graph, id, node, &cache, [u, v]) {
                    cache.insert((id, i), val);
                }
            }
            k if k.is_color_extra() => {
                for (i, val) in eval_color_node(graph, id, node, &cache) {
                    cache.insert((id, i), val);
                }
            }
            k if k.is_texture_extra() => {
                let uv = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Vector([u, v, 0.0]))
                    .as_vector();
                let c = texture_node_color(node, textures, uv[0], uv[1]);
                cache.insert((id, 0), SocketVal::Color(c));
                cache.insert((id, 1), SocketVal::Scalar(SocketVal::Color(c).as_scalar()));
            }
            k if k.is_shader_extra() && !k.is_volume_shader() => {
                let c = match k {
                    ShaderNodeKind::AddShader | ShaderNodeKind::MixShader => {
                        let (ia, ib) = if k == ShaderNodeKind::MixShader { (1, 2) } else { (0, 1) };
                        let fac = if k == ShaderNodeKind::MixShader {
                            find_input(graph, id, 0, &cache)
                                .map(|v| v.as_scalar())
                                .unwrap_or(node.value)
                        } else {
                            1.0
                        };
                        let a = find_input(graph, id, ia, &cache)
                            .unwrap_or(SocketVal::Color([0.0, 0.0, 0.0, 1.0]))
                            .as_color();
                        let b = find_input(graph, id, ib, &cache)
                            .unwrap_or(SocketVal::Color([0.0, 0.0, 0.0, 1.0]))
                            .as_color();
                        if k == ShaderNodeKind::MixShader {
                            mix4(a, b, fac)
                        } else {
                            [(a[0] + b[0]).min(1.0), (a[1] + b[1]).min(1.0), (a[2] + b[2]).min(1.0), 1.0]
                        }
                    }
                    ShaderNodeKind::LightAccumulation | ShaderNodeKind::Holdout | ShaderNodeKind::RayPortalBsdf => {
                        [0.0, 0.0, 0.0, 1.0]
                    }
                    _ => find_input(graph, id, 0, &cache)
                        .unwrap_or(SocketVal::Color(node.color))
                        .as_color(),
                };
                cache.insert((id, 0), SocketVal::Color(c));
            }
            ShaderNodeKind::Output => {
                if let Some(s) = find_input(graph, id, 0, &cache) {
                    cache.insert((id, 0), s);
                }
            }
            _ => {}
        }
    }
    // Prefer Output surface, then Principled, then base eval.
    for n in &graph.nodes {
        if n.kind == ShaderNodeKind::Output {
            if let Some(v) = cache.get(&(n.id, 0)) {
                return v.as_color();
            }
        }
    }
    for n in &graph.nodes {
        if n.kind == ShaderNodeKind::Principled {
            if let Some(v) = cache.get(&(n.id, 0)) {
                return v.as_color();
            }
        }
    }
    base.base_color
}
