//! Serializable material shader graph and CPU evaluation into PBR slots / textures.
//! Node set aligned with Blender Shader Editor categories (Input, Output, Shader, Texture, Color, Vector/Utilities).
use crate::material::{Material, TextureData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SocketType {
    Float,
    Color,
    Vector,
    Shader,
    String,
}

#[derive(Clone, Copy, Debug)]
pub struct SocketDef {
    pub name: &'static str,
    pub ty: SocketType,
    pub input: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShaderNodeKind {
    // Output
    Output,
    LightOutput,
    WorldOutput,
    AovOutput,
    // Shader
    Principled,
    Diffuse,
    Glossy,
    Emission,
    AddShader,
    MixShader,
    Background,
    Glass,
    Hair,
    Holdout,
    Metallic,
    PrincipledHair,
    PrincipledVolume,
    RayPortal,
    Refraction,
    Specular,
    Subsurface,
    Toon,
    Translucent,
    Transparent,
    Sheen,
    VolumeAbsorption,
    VolumeScatter,
    VolumeCoefficients,
    LightAccumulation,
    // Texture
    ImageTexture,
    Noise,
    Voronoi,
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
    // Color
    ColorRamp,
    Mix,
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
    // Vector / Utilities
    Mapping,
    Bump,
    NormalMap,
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
    // Input
    Rgb,
    Value,
    TexCoord,
    UvMap,
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
    ColorAttribute,
    VolumeInfo,
    Wireframe,
}

impl ShaderNodeKind {
    pub fn title(self) -> &'static str {
        use ShaderNodeKind::*;
        match self {
            Output => "Material Output",
            LightOutput => "Light Output",
            WorldOutput => "World Output",
            AovOutput => "AOV Output",
            Principled => "Principled BSDF",
            Diffuse => "Diffuse BSDF",
            Glossy => "Glossy BSDF",
            Emission => "Emission",
            AddShader => "Add Shader",
            MixShader => "Mix Shader",
            Background => "Background",
            Glass => "Glass BSDF",
            Hair => "Hair BSDF",
            Holdout => "Holdout",
            Metallic => "Metallic BSDF",
            PrincipledHair => "Principled Hair BSDF",
            PrincipledVolume => "Principled Volume",
            RayPortal => "Ray Portal BSDF",
            Refraction => "Refraction BSDF",
            Specular => "Specular BSDF",
            Subsurface => "Subsurface Scattering",
            Toon => "Toon BSDF",
            Translucent => "Translucent BSDF",
            Transparent => "Transparent BSDF",
            Sheen => "Sheen BSDF",
            VolumeAbsorption => "Volume Absorption",
            VolumeScatter => "Volume Scatter",
            VolumeCoefficients => "Volume Coefficients",
            LightAccumulation => "Light Accumulation",
            ImageTexture => "Image Texture",
            Noise => "Noise Texture",
            Voronoi => "Voronoi Texture",
            Brick => "Brick Texture",
            Checker => "Checker Texture",
            Environment => "Environment Texture",
            Gabor => "Gabor Texture",
            Gradient => "Gradient Texture",
            Ies => "IES Texture",
            Magic => "Magic Texture",
            Sky => "Sky Texture",
            Wave => "Wave Texture",
            WhiteNoise => "White Noise Texture",
            ColorRamp => "Color Ramp",
            Mix => "Mix",
            Blackbody => "Blackbody",
            BrightContrast => "Brightness/Contrast",
            Gamma => "Gamma",
            HueSatVal => "Hue/Saturation/Value",
            InvertColor => "Invert Color",
            LightFalloff => "Light Falloff",
            RgbCurves => "RGB Curves",
            Wavelength => "Wavelength",
            CombineColor => "Combine Color",
            SeparateColor => "Separate Color",
            RgbToBw => "RGB to BW",
            ShaderToRgb => "Shader to RGB",
            Mapping => "Mapping",
            Bump => "Bump",
            NormalMap => "Normal Map",
            CombineXyz => "Combine XYZ",
            SeparateXyz => "Separate XYZ",
            VectorMath => "Vector Math",
            VectorRotate => "Vector Rotate",
            VectorTransform => "Vector Transform",
            VectorCurves => "Vector Curves",
            Normal => "Normal",
            Displacement => "Displacement",
            VectorDisplacement => "Vector Displacement",
            Math => "Math",
            MapRange => "Map Range",
            Clamp => "Clamp",
            FloatCurve => "Float Curve",
            Switch => "Switch",
            IndexSwitch => "Index Switch",
            MenuSwitch => "Menu Switch",
            Script => "Script",
            Group => "Group",
            Rgb => "RGB",
            Value => "Value",
            TexCoord => "Texture Coordinate",
            UvMap => "UV Map",
            AmbientOcclusion => "Ambient Occlusion",
            Attribute => "Attribute",
            Bevel => "Bevel",
            CameraData => "Camera Data",
            Fresnel => "Fresnel",
            Geometry => "Geometry",
            CurvesInfo => "Curves Info",
            LayerWeight => "Layer Weight",
            LightEvaluation => "Light Evaluation",
            LightInfo => "Light Info",
            LightPath => "Light Path",
            ObjectInfo => "Object Info",
            ParticleInfo => "Particle Info",
            PointInfo => "Point Info",
            Raycast => "Raycast",
            SceneTime => "Scene Time",
            ShadowRaycast => "Shadow Raycast",
            Tangent => "Tangent",
            ColorAttribute => "Color Attribute",
            VolumeInfo => "Volume Info",
            Wireframe => "Wireframe",
        }
    }

    pub fn category(self) -> &'static str {
        use ShaderNodeKind::*;
        match self {
            Output | LightOutput | WorldOutput | AovOutput => "Output",
            Principled | Diffuse | Glossy | Emission | AddShader | MixShader | Background
            | Glass | Hair | Holdout | Metallic | PrincipledHair | PrincipledVolume | RayPortal
            | Refraction | Specular | Subsurface | Toon | Translucent | Transparent | Sheen
            | VolumeAbsorption | VolumeScatter | VolumeCoefficients | LightAccumulation => "Shader",
            ImageTexture | Noise | Voronoi | Brick | Checker | Environment | Gabor | Gradient
            | Ies | Magic | Sky | Wave | WhiteNoise => "Texture",
            ColorRamp | Mix | Blackbody | BrightContrast | Gamma | HueSatVal | InvertColor
            | LightFalloff | RgbCurves | Wavelength | CombineColor | SeparateColor | RgbToBw
            | ShaderToRgb => "Color",
            Mapping | Bump | NormalMap | CombineXyz | SeparateXyz | VectorMath | VectorRotate
            | VectorTransform | VectorCurves | Normal | Displacement | VectorDisplacement
            | Math | MapRange | Clamp | FloatCurve | Switch | IndexSwitch | MenuSwitch | Script
            | Group => "Vector",
            _ => "Input",
        }
    }

    /// Primary inputs then outputs. Socket index = order within inputs or within outputs separately
    /// for UI; for links, from_socket indexes outputs, to_socket indexes inputs.
    pub fn inputs(self) -> &'static [SocketDef] {
        use ShaderNodeKind::*;
        use SocketType::*;
        match self {
            Output => &[
                SocketDef {
                    name: "Surface",
                    ty: Shader,
                    input: true,
                },
                SocketDef {
                    name: "Volume",
                    ty: Shader,
                    input: true,
                },
                SocketDef {
                    name: "Displacement",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Thickness",
                    ty: Float,
                    input: true,
                },
            ],
            LightOutput | WorldOutput => &[
                SocketDef {
                    name: "Surface",
                    ty: Shader,
                    input: true,
                },
                SocketDef {
                    name: "Volume",
                    ty: Shader,
                    input: true,
                },
            ],
            AovOutput => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Value",
                    ty: Float,
                    input: true,
                },
            ],
            Principled => &[
                SocketDef {
                    name: "Base Color",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Metallic",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Roughness",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "IOR",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Alpha",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Normal",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Emission Color",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Emission Strength",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Coat Weight",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Sheen Weight",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Subsurface Weight",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Transmission Weight",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Specular IOR Level",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "AO",
                    ty: Float,
                    input: true,
                },
            ],
            Diffuse => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Roughness",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Normal",
                    ty: Vector,
                    input: true,
                },
            ],
            Glossy => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Roughness",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Anisotropy",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Rotation",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Normal",
                    ty: Vector,
                    input: true,
                },
            ],
            Emission | Background => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Strength",
                    ty: Float,
                    input: true,
                },
            ],
            AddShader => &[
                SocketDef {
                    name: "Shader",
                    ty: Shader,
                    input: true,
                },
                SocketDef {
                    name: "Shader",
                    ty: Shader,
                    input: true,
                },
            ],
            MixShader => &[
                SocketDef {
                    name: "Fac",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Shader",
                    ty: Shader,
                    input: true,
                },
                SocketDef {
                    name: "Shader",
                    ty: Shader,
                    input: true,
                },
            ],
            Glass | Refraction => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Roughness",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "IOR",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Normal",
                    ty: Vector,
                    input: true,
                },
            ],
            Transparent | Translucent | Sheen | Holdout => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Normal",
                    ty: Vector,
                    input: true,
                },
            ],
            VolumeAbsorption | VolumeScatter | VolumeCoefficients | PrincipledVolume => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Density",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Anisotropy",
                    ty: Float,
                    input: true,
                },
            ],
            ImageTexture => &[SocketDef {
                name: "Vector",
                ty: Vector,
                input: true,
            }],
            Noise => &[
                SocketDef {
                    name: "Vector",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "W",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Scale",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Detail",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Roughness",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Lacunarity",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Distortion",
                    ty: Float,
                    input: true,
                },
            ],
            Voronoi => &[
                SocketDef {
                    name: "Vector",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "W",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Scale",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Detail",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Roughness",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Lacunarity",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Smoothness",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Randomness",
                    ty: Float,
                    input: true,
                },
            ],
            Brick => &[
                SocketDef {
                    name: "Vector",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Color1",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Color2",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Mortar",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Scale",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Mortar Size",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Mortar Smooth",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Bias",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Brick Width",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Row Height",
                    ty: Float,
                    input: true,
                },
            ],
            Checker => &[
                SocketDef {
                    name: "Vector",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Color1",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Color2",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Scale",
                    ty: Float,
                    input: true,
                },
            ],
            Environment => &[SocketDef {
                name: "Vector",
                ty: Vector,
                input: true,
            }],
            Gabor => &[
                SocketDef {
                    name: "Vector",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Scale",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Frequency",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Anisotropy",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Orientation",
                    ty: Float,
                    input: true,
                },
            ],
            Gradient => &[SocketDef {
                name: "Vector",
                ty: Vector,
                input: true,
            }],
            Ies => &[
                SocketDef {
                    name: "Vector",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Strength",
                    ty: Float,
                    input: true,
                },
            ],
            Magic => &[
                SocketDef {
                    name: "Vector",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Scale",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Distortion",
                    ty: Float,
                    input: true,
                },
            ],
            Sky => &[SocketDef {
                name: "Vector",
                ty: Vector,
                input: true,
            }],
            Wave => &[
                SocketDef {
                    name: "Vector",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Scale",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Distortion",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Detail",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Detail Scale",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Detail Roughness",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Phase Offset",
                    ty: Float,
                    input: true,
                },
            ],
            WhiteNoise => &[
                SocketDef {
                    name: "Vector",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "W",
                    ty: Float,
                    input: true,
                },
            ],
            ColorRamp => &[SocketDef {
                name: "Fac",
                ty: Float,
                input: true,
            }],
            Mix => &[
                SocketDef {
                    name: "Fac",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "A",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "B",
                    ty: Color,
                    input: true,
                },
            ],
            Blackbody | Wavelength => &[SocketDef {
                name: "Temperature",
                ty: Float,
                input: true,
            }],
            BrightContrast => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Bright",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Contrast",
                    ty: Float,
                    input: true,
                },
            ],
            Gamma => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Gamma",
                    ty: Float,
                    input: true,
                },
            ],
            HueSatVal => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Hue",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Saturation",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Value",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Fac",
                    ty: Float,
                    input: true,
                },
            ],
            InvertColor => &[
                SocketDef {
                    name: "Fac",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: true,
                },
            ],
            LightFalloff => &[
                SocketDef {
                    name: "Strength",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Smooth",
                    ty: Float,
                    input: true,
                },
            ],
            RgbCurves | RgbToBw | ShaderToRgb => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Fac",
                    ty: Float,
                    input: true,
                },
            ],
            CombineColor | CombineXyz => &[
                SocketDef {
                    name: "R/X",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "G/Y",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "B/Z",
                    ty: Float,
                    input: true,
                },
            ],
            SeparateColor | SeparateXyz => &[SocketDef {
                name: "Color/Vector",
                ty: Color,
                input: true,
            }],
            Mapping => &[
                SocketDef {
                    name: "Vector",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Location",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Rotation",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Scale",
                    ty: Vector,
                    input: true,
                },
            ],
            Bump => &[
                SocketDef {
                    name: "Strength",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Distance",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Height",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Normal",
                    ty: Vector,
                    input: true,
                },
            ],
            NormalMap => &[
                SocketDef {
                    name: "Strength",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: true,
                },
            ],
            VectorMath => &[
                SocketDef {
                    name: "Vector",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Vector",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Scale",
                    ty: Float,
                    input: true,
                },
            ],
            VectorRotate => &[
                SocketDef {
                    name: "Vector",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Center",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Axis",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Angle",
                    ty: Float,
                    input: true,
                },
            ],
            Math => &[
                SocketDef {
                    name: "Value",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Value",
                    ty: Float,
                    input: true,
                },
            ],
            MapRange => &[
                SocketDef {
                    name: "Value",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "From Min",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "From Max",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "To Min",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "To Max",
                    ty: Float,
                    input: true,
                },
            ],
            Clamp => &[
                SocketDef {
                    name: "Value",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Min",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Max",
                    ty: Float,
                    input: true,
                },
            ],
            Switch | IndexSwitch | MenuSwitch => &[
                SocketDef {
                    name: "Switch",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "False",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "True",
                    ty: Color,
                    input: true,
                },
            ],
            FloatCurve | VectorCurves | VectorTransform | Displacement | VectorDisplacement
            | Normal => &[
                SocketDef {
                    name: "Value",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Vector",
                    ty: Vector,
                    input: true,
                },
            ],
            Script => &[
                SocketDef {
                    name: "a",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "b",
                    ty: Float,
                    input: true,
                },
            ],
            Fresnel => &[
                SocketDef {
                    name: "IOR",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Normal",
                    ty: Vector,
                    input: true,
                },
            ],
            LayerWeight => &[
                SocketDef {
                    name: "Blend",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Normal",
                    ty: Vector,
                    input: true,
                },
            ],
            AmbientOcclusion => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: true,
                },
                SocketDef {
                    name: "Distance",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Normal",
                    ty: Vector,
                    input: true,
                },
            ],
            Bevel => &[
                SocketDef {
                    name: "Radius",
                    ty: Float,
                    input: true,
                },
                SocketDef {
                    name: "Normal",
                    ty: Vector,
                    input: true,
                },
            ],
            Raycast | ShadowRaycast => &[
                SocketDef {
                    name: "Position",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Direction",
                    ty: Vector,
                    input: true,
                },
                SocketDef {
                    name: "Length",
                    ty: Float,
                    input: true,
                },
            ],
            Wireframe => &[SocketDef {
                name: "Size",
                ty: Float,
                input: true,
            }],
            _ => &[],
        }
    }

    pub fn outputs(self) -> &'static [SocketDef] {
        use ShaderNodeKind::*;
        use SocketType::*;
        match self {
            Output | LightOutput | WorldOutput | AovOutput => &[],
            Principled | Diffuse | Glossy | Emission | AddShader | MixShader | Background
            | Glass | Hair | Holdout | Metallic | PrincipledHair | PrincipledVolume | RayPortal
            | Refraction | Specular | Subsurface | Toon | Translucent | Transparent | Sheen
            | VolumeAbsorption | VolumeScatter | VolumeCoefficients | LightAccumulation => {
                &[SocketDef {
                    name: "BSDF",
                    ty: Shader,
                    input: false,
                }]
            }
            ImageTexture | Environment => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: false,
                },
                SocketDef {
                    name: "Alpha",
                    ty: Float,
                    input: false,
                },
            ],
            Noise | Wave | Magic | Gradient | Gabor | Ies => &[
                SocketDef {
                    name: "Fac",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: false,
                },
            ],
            WhiteNoise => &[
                SocketDef {
                    name: "Value",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: false,
                },
            ],
            Voronoi => &[
                SocketDef {
                    name: "Distance",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: false,
                },
                SocketDef {
                    name: "Position",
                    ty: Vector,
                    input: false,
                },
                SocketDef {
                    name: "W",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "Radius",
                    ty: Float,
                    input: false,
                },
            ],
            Brick | Checker => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: false,
                },
                SocketDef {
                    name: "Fac",
                    ty: Float,
                    input: false,
                },
            ],
            Sky => &[SocketDef {
                name: "Color",
                ty: Color,
                input: false,
            }],
            ColorRamp | Mix | Blackbody | BrightContrast | Gamma | HueSatVal | InvertColor
            | RgbCurves | Wavelength | Rgb | ColorAttribute => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: false,
                },
                SocketDef {
                    name: "Alpha",
                    ty: Float,
                    input: false,
                },
            ],
            RgbToBw | Value | Math | MapRange | Clamp | FloatCurve | Fresnel | Wireframe
            | AmbientOcclusion | SceneTime => &[SocketDef {
                name: "Value",
                ty: Float,
                input: false,
            }],
            SeparateColor | SeparateXyz => &[
                SocketDef {
                    name: "R/X",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "G/Y",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "B/Z",
                    ty: Float,
                    input: false,
                },
            ],
            CombineColor | CombineXyz | Mapping | VectorMath | VectorRotate | VectorTransform
            | VectorCurves | Normal | Displacement | VectorDisplacement | TexCoord | UvMap
            | Tangent | Geometry | Bevel | NormalMap | Bump => &[SocketDef {
                name: "Vector",
                ty: Vector,
                input: false,
            }],
            LayerWeight => &[
                SocketDef {
                    name: "Fresnel",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "Facing",
                    ty: Float,
                    input: false,
                },
            ],
            CameraData => &[
                SocketDef {
                    name: "View Vector",
                    ty: Vector,
                    input: false,
                },
                SocketDef {
                    name: "View Z Depth",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "View Distance",
                    ty: Float,
                    input: false,
                },
            ],
            LightPath => &[
                SocketDef {
                    name: "Is Camera Ray",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "Is Shadow Ray",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "Is Diffuse Ray",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "Is Glossy Ray",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "Ray Depth",
                    ty: Float,
                    input: false,
                },
            ],
            ObjectInfo => &[
                SocketDef {
                    name: "Location",
                    ty: Vector,
                    input: false,
                },
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: false,
                },
                SocketDef {
                    name: "Object Index",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "Material Index",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "Random",
                    ty: Float,
                    input: false,
                },
            ],
            ShaderToRgb => &[
                SocketDef {
                    name: "Color",
                    ty: Color,
                    input: false,
                },
                SocketDef {
                    name: "Alpha",
                    ty: Float,
                    input: false,
                },
            ],
            LightFalloff => &[
                SocketDef {
                    name: "Quadratic",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "Linear",
                    ty: Float,
                    input: false,
                },
                SocketDef {
                    name: "Constant",
                    ty: Float,
                    input: false,
                },
            ],
            Switch | IndexSwitch | MenuSwitch | Group | Script => &[SocketDef {
                name: "Output",
                ty: Color,
                input: false,
            }],
            _ => &[SocketDef {
                name: "Value",
                ty: Float,
                input: false,
            }],
        }
    }

    pub fn is_input_extra(self) -> bool {
        self.category() == "Input" && !matches!(self, Self::Rgb | Self::Value | Self::TexCoord)
    }

    pub fn all_kinds() -> &'static [ShaderNodeKind] {
        use ShaderNodeKind::*;
        &[
            Output,
            LightOutput,
            WorldOutput,
            AovOutput,
            Principled,
            Diffuse,
            Glossy,
            Emission,
            AddShader,
            MixShader,
            Background,
            Glass,
            Hair,
            Holdout,
            Metallic,
            PrincipledHair,
            PrincipledVolume,
            RayPortal,
            Refraction,
            Specular,
            Subsurface,
            Toon,
            Translucent,
            Transparent,
            Sheen,
            VolumeAbsorption,
            VolumeScatter,
            VolumeCoefficients,
            LightAccumulation,
            ImageTexture,
            Noise,
            Voronoi,
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
            ColorRamp,
            Mix,
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
            Mapping,
            Bump,
            NormalMap,
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
            Rgb,
            Value,
            TexCoord,
            UvMap,
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
            ColorAttribute,
            VolumeInfo,
            Wireframe,
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShaderNode {
    pub id: u64,
    pub kind: ShaderNodeKind,
    pub pos: [f32; 2],
    pub image_index: Option<usize>,
    /// Primary scalar (legacy + defaults for first float prop).
    pub value: f32,
    /// RGBA 0–1 for RGB / Emission color nodes.
    pub color: [f32; 4],
    /// Extra float properties (Scale axes, Math mode, etc.).
    #[serde(default)]
    pub props: [f32; 16],
    /// Per-input defaults, indexed by input socket. Components are interpreted
    /// according to the socket type and kept separate from legacy node props.
    #[serde(default)]
    pub input_defaults: Vec<[f32; 4]>,
    /// Mode enum (blend mode, math operation, color mode, …).
    #[serde(default)]
    pub mode: u8,
    /// Name property (Attribute, UV Map, AOV, Script expression).
    #[serde(default)]
    pub name_prop: String,
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
            props: [0.0; 16],
            input_defaults: Vec::new(),
            mode: 0,
            name_prop: String::new(),
        }
    }
}

impl ShaderNode {
    pub fn uses_implicit_coordinates(&self, socket: usize) -> bool {
        socket == 0
            && matches!(
                self.kind,
                ShaderNodeKind::ImageTexture
                    | ShaderNodeKind::Environment
                    | ShaderNodeKind::Noise
                    | ShaderNodeKind::Voronoi
                    | ShaderNodeKind::Brick
                    | ShaderNodeKind::Checker
                    | ShaderNodeKind::Gradient
                    | ShaderNodeKind::Wave
                    | ShaderNodeKind::Magic
                    | ShaderNodeKind::Gabor
                    | ShaderNodeKind::WhiteNoise
                    | ShaderNodeKind::Mapping
            )
            && self.input_defaults.get(socket).map_or(true, |value| {
                value[0] == 0.0 && value[1] == 0.0 && value[2] == 0.0
            })
    }

    pub fn with_kind(id: u64, kind: ShaderNodeKind, pos: [f32; 2]) -> Self {
        let mut node = Self {
            id,
            kind,
            pos,
            ..Self::default()
        };
        if matches!(kind, ShaderNodeKind::Noise | ShaderNodeKind::WhiteNoise) {
            node.mode = 2;
        }
        if kind == ShaderNodeKind::Brick {
            node.props[..4].copy_from_slice(&[0.5, 2.0, 1.0, 2.0]);
        }
        if kind == ShaderNodeKind::Magic {
            node.props[8] = 2.0;
            node.props[9] = 1.0;
        }
        node.input_defaults = kind
            .inputs()
            .iter()
            .map(|socket| {
                let mut value = [0.0; 4];
                match (kind, socket.name) {
                    (ShaderNodeKind::Principled, "Base Color") => value = node.color,
                    (ShaderNodeKind::Principled, "Roughness") => value[0] = 0.5,
                    (ShaderNodeKind::Principled, "IOR") => value[0] = 1.45,
                    (ShaderNodeKind::Principled, "Alpha") => value[0] = 1.0,
                    (ShaderNodeKind::Principled, "AO") => value[0] = 1.0,
                    (ShaderNodeKind::Principled, "Emission Color") => value = [0.0, 0.0, 0.0, 1.0],
                    (ShaderNodeKind::Principled, "Normal") => value = [0.0, 0.0, 1.0, 0.0],
                    (ShaderNodeKind::Noise, "Scale")
                    | (ShaderNodeKind::Voronoi, "Scale")
                    | (ShaderNodeKind::Wave, "Scale")
                    | (ShaderNodeKind::Magic, "Scale")
                    | (ShaderNodeKind::Brick, "Scale")
                    | (ShaderNodeKind::Checker, "Scale") => value[0] = 5.0,
                    (ShaderNodeKind::Noise, "Detail") => value[0] = 2.0,
                    (ShaderNodeKind::Noise, "Roughness") => value[0] = 0.5,
                    (ShaderNodeKind::Noise, "Lacunarity") => value[0] = 2.0,
                    (ShaderNodeKind::Voronoi, "Randomness") => value[0] = 1.0,
                    (ShaderNodeKind::Voronoi, "Smoothness") => value[0] = 0.5,
                    (ShaderNodeKind::Wave, "Detail") => value[0] = 2.0,
                    (ShaderNodeKind::Wave, "Detail Scale") => value[0] = 1.0,
                    (ShaderNodeKind::Wave, "Detail Roughness") => value[0] = 0.5,
                    (ShaderNodeKind::Magic, "Distortion") => value[0] = 1.0,
                    (ShaderNodeKind::Brick, "Color1") => value = [0.8, 0.8, 0.8, 1.0],
                    (ShaderNodeKind::Brick, "Color2") => value = [0.2, 0.2, 0.2, 1.0],
                    (ShaderNodeKind::Brick, "Mortar") => value = [0.0, 0.0, 0.0, 1.0],
                    (ShaderNodeKind::Brick, "Mortar Size") => value[0] = 0.02,
                    (ShaderNodeKind::Brick, "Mortar Smooth") => value[0] = 0.1,
                    (ShaderNodeKind::Brick, "Brick Width") => value[0] = 0.5,
                    (ShaderNodeKind::Brick, "Row Height") => value[0] = 0.25,
                    (ShaderNodeKind::Checker, "Color1") => value = [1.0, 1.0, 1.0, 1.0],
                    (ShaderNodeKind::Checker, "Color2") => value = [0.0, 0.0, 0.0, 1.0],
                    (ShaderNodeKind::Gabor, "Scale") => value[0] = 5.0,
                    (ShaderNodeKind::Gabor, "Frequency") => value[0] = 2.0,
                    (ShaderNodeKind::Gabor, "Anisotropy") => value[0] = 1.0,
                    (ShaderNodeKind::Mapping, "Scale") => value = [1.0; 4],
                    (ShaderNodeKind::Bump, "Strength") => value[0] = 1.0,
                    (ShaderNodeKind::Bump, "Distance") => value[0] = 1.0,
                    (ShaderNodeKind::Bump, "Normal") => value = [0.0, 0.0, 1.0, 0.0],
                    (ShaderNodeKind::NormalMap, "Color") => value = [0.5, 0.5, 1.0, 1.0],
                    (ShaderNodeKind::Math, "Value") => value[0] = node.value,
                    (ShaderNodeKind::Emission | ShaderNodeKind::Background, "Strength") => {
                        value[0] = 1.0
                    }
                    (ShaderNodeKind::ColorRamp, "Fac") => value[0] = 0.5,
                    (ShaderNodeKind::Mix, "Fac") => value[0] = 0.5,
                    (ShaderNodeKind::Mix, "B") => value = [1.0; 4],
                    _ => {}
                }
                value
            })
            .collect();
        node
    }

    pub fn socket_default(&self, socket: usize) -> [f32; 4] {
        self.input_defaults.get(socket).copied().unwrap_or_else(|| {
            let Some(def) = self.kind.inputs().get(socket) else {
                return [0.0; 4];
            };
            if self.kind == ShaderNodeKind::Principled {
                return match socket {
                    0 => self.color,
                    1 => [self.props[0], 0.0, 0.0, 0.0],
                    2 => [
                        if self.props[1] == 0.0 {
                            0.5
                        } else {
                            self.props[1]
                        },
                        0.0,
                        0.0,
                        0.0,
                    ],
                    3 => [
                        if self.props[2] == 0.0 {
                            1.45
                        } else {
                            self.props[2]
                        },
                        0.0,
                        0.0,
                        0.0,
                    ],
                    4 => [
                        if self.props[3] == 0.0 {
                            1.0
                        } else {
                            self.props[3]
                        },
                        0.0,
                        0.0,
                        0.0,
                    ],
                    6 => [0.0, 0.0, 0.0, 1.0],
                    7 => [0.0, 0.0, 0.0, 0.0],
                    13 => [1.0, 0.0, 0.0, 0.0],
                    _ => [0.0; 4],
                };
            }
            if self.kind == ShaderNodeKind::Noise {
                return match socket {
                    0 => [0.5, 0.5, 0.0, 0.0],
                    1 => [self.props[5], 0.0, 0.0, 0.0],
                    2 => [
                        if (self.value - 0.5).abs() < 1e-6 {
                            5.0
                        } else {
                            self.value
                        },
                        0.0,
                        0.0,
                        0.0,
                    ],
                    3 => [
                        if self.props[0] == 0.0 {
                            2.0
                        } else {
                            self.props[0]
                        },
                        0.0,
                        0.0,
                        0.0,
                    ],
                    4 => [
                        if self.props[1] == 0.0 {
                            0.5
                        } else {
                            self.props[1]
                        },
                        0.0,
                        0.0,
                        0.0,
                    ],
                    5 => [
                        if self.props[2] == 0.0 {
                            2.0
                        } else {
                            self.props[2]
                        },
                        0.0,
                        0.0,
                        0.0,
                    ],
                    6 => [self.props[3], 0.0, 0.0, 0.0],
                    _ => [0.0; 4],
                };
            }
            if self.kind == ShaderNodeKind::Voronoi {
                return match socket {
                    0 => [0.5, 0.5, 0.0, 0.0],
                    2 => [
                        if (self.value - 0.5).abs() < 1e-6 {
                            5.0
                        } else {
                            self.value
                        },
                        0.0,
                        0.0,
                        0.0,
                    ],
                    7 => [
                        if self.props[0] == 0.0 {
                            1.0
                        } else {
                            self.props[0]
                        },
                        0.0,
                        0.0,
                        0.0,
                    ],
                    _ => [self.props[socket.min(15)], 0.0, 0.0, 0.0],
                };
            }
            if self.kind == ShaderNodeKind::Mapping {
                return match socket {
                    0 => [0.5, 0.5, 0.0, 0.0],
                    1 | 2 => [0.0; 4],
                    3 => [
                        if self.value == 0.0 { 1.0 } else { self.value },
                        if self.props[0] == 0.0 {
                            1.0
                        } else {
                            self.props[0]
                        },
                        if self.props[1] == 0.0 {
                            1.0
                        } else {
                            self.props[1]
                        },
                        0.0,
                    ],
                    _ => [0.0; 4],
                };
            }
            match def.ty {
                SocketType::Color | SocketType::Shader => self.color,
                SocketType::Vector => {
                    let start = socket.saturating_mul(3).min(13);
                    [
                        self.props[start],
                        self.props[(start + 1).min(15)],
                        self.props[(start + 2).min(15)],
                        0.0,
                    ]
                }
                _ => {
                    let value = match (self.kind, def.name) {
                        (
                            ShaderNodeKind::Wave | ShaderNodeKind::Magic | ShaderNodeKind::Gabor,
                            "Scale",
                        ) => {
                            if (self.value - 0.5).abs() < 1e-6 {
                                5.0
                            } else {
                                self.value
                            }
                        }
                        (ShaderNodeKind::Brick, "Scale") | (ShaderNodeKind::Checker, "Scale") => {
                            if self.value == 0.0 {
                                5.0
                            } else {
                                self.value
                            }
                        }
                        (ShaderNodeKind::Emission | ShaderNodeKind::Background, "Strength") => 1.0,
                        (ShaderNodeKind::Math, _) => self.value,
                        _ => self.value,
                    };
                    [value, 0.0, 0.0, 0.0]
                }
            }
        })
    }

    pub fn set_socket_default(&mut self, socket: usize, value: [f32; 4]) {
        let count = self.kind.inputs().len();
        if self.input_defaults.len() < count {
            let old = (0..count).map(|i| self.socket_default(i)).collect();
            self.input_defaults = old;
        }
        if let Some(slot) = self.input_defaults.get_mut(socket) {
            *slot = value;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShaderLink {
    pub from: u64,
    pub to: u64,
    #[serde(default)]
    pub from_socket: u8,
    #[serde(default)]
    pub to_socket: u8,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
                    value: 1.45, // IOR default stored
                    props: {
                        let mut p = [0.0; 16];
                        p[0] = 0.0; // metallic
                        p[1] = 0.5; // roughness
                        p[2] = 1.45; // ior
                        p[3] = 1.0; // alpha
                        p
                    },
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
    pub new_textures: Vec<TextureData>,
    pub displacement_scale: f32,
    pub volume_density: f32,
    pub ior: f32,
    pub alpha: f32,
    pub emission_strength: f32,
}

fn hash2(x: f32, y: f32) -> f32 {
    let mut n = x * 127.1 + y * 311.7;
    n = (n.sin() * 43758.5453).rem_euclid(1.0);
    n
}

fn noise2(x: f32, y: f32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let fx = x.rem_euclid(1.0);
    let fy = y.rem_euclid(1.0);
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let a = hash2(ix, iy);
    let b = hash2(ix + 1.0, iy);
    let c = hash2(ix, iy + 1.0);
    let d = hash2(ix + 1.0, iy + 1.0);
    a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy
}

fn hash3(x: f32, y: f32, z: f32) -> f32 {
    hash2(hash2(x, y) * 4096.0, z)
}

fn hash4(x: f32, y: f32, z: f32, w: f32) -> f32 {
    hash2(hash3(x, y, z) * 4096.0, w)
}

fn white_hash(seed: [f32; 4], dimension: u8) -> f32 {
    let mut hash = seed[0].to_bits().wrapping_mul(374761393);
    if dimension >= 1 {
        hash = hash.wrapping_add(seed[1].to_bits().wrapping_mul(668265263));
    }
    if dimension >= 2 {
        hash = hash.wrapping_add(seed[2].to_bits().wrapping_mul(2246822519));
    }
    if dimension >= 3 {
        hash = hash.wrapping_add(seed[3].to_bits().wrapping_mul(3266489917));
    }
    hash = (hash ^ (hash >> 13)).wrapping_mul(1274126177);
    hash ^= hash >> 16;
    (hash & 0x00FF_FFFF) as f32 / 16_777_216.0
}

fn white_noise_texture(p: [f32; 3], w: f32, dimension: u8) -> (f32, [f32; 4]) {
    let sample = |channel: usize| {
        let channel = channel as f32;
        let seed = [
            p[0] + channel * 17.13,
            p[1] + channel * 29.71,
            p[2] + channel * 47.83,
            w + channel * 61.37,
        ];
        let dimensions = match dimension.min(3) {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 3,
        };
        let seed = if dimensions == 0 {
            [seed[3], 0.0, 0.0, 0.0]
        } else {
            seed
        };
        white_hash(seed, dimensions)
    };
    let value = sample(0);
    (
        value,
        [sample(1), sample(2), sample(3), 1.0],
    )
}

fn sample_tex(tex: &TextureData, u: f32, v: f32) -> [f32; 4] {
    sample_tex_with_color_space(tex, u, v, true)
}

fn sample_tex_data(tex: &TextureData, u: f32, v: f32) -> [f32; 4] {
    sample_tex_with_color_space(tex, u, v, false)
}

fn sample_tex_with_color_space(tex: &TextureData, u: f32, v: f32, srgb: bool) -> [f32; 4] {
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
    let decode = |byte: u8| {
        let value = byte as f32 / 255.0;
        if srgb {
            srgb_to_linear(value)
        } else {
            value
        }
    };
    [
        decode(tex.rgba8[i]),
        decode(tex.rgba8[i + 1]),
        decode(tex.rgba8[i + 2]),
        tex.rgba8[i + 3] as f32 / 255.0,
    ]
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let encoded = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

fn rotate_euler_xyz(v: [f32; 3], rotation: [f32; 3]) -> [f32; 3] {
    let (sx, cx) = rotation[0].sin_cos();
    let (sy, cy) = rotation[1].sin_cos();
    let (sz, cz) = rotation[2].sin_cos();
    let x = [v[0], v[1] * cx - v[2] * sx, v[1] * sx + v[2] * cx];
    let y = [x[0] * cy + x[2] * sy, x[1], -x[0] * sy + x[2] * cy];
    [y[0] * cz - y[1] * sz, y[0] * sz + y[1] * cz, y[2]]
}

/// Max stops stored on a ColorRamp node (fits in props + color).
pub const COLOR_RAMP_MAX_STOPS: usize = 4;

/// Unpack ColorRamp stops from a ShaderNode.
/// Layout (Blender-like):
///   color          = stop 0 RGBA
///   props[0..3]    = stop 1 RGBA
///   props[4..7]    = stop 2 RGBA
///   props[8..11]   = stop 3 RGBA
///   props[12..15]  = positions 0..3
///   value          = stop count (2..=4), default 2
///   mode low 4 bits = interpolation: 0 Linear, 1 Ease, 2 Constant, 3 Cardinal, 4 B-Spline
///   mode bits 4..5  = color mode: 0 RGB, 1 HSV, 2 HSL (RGB used for eval for now)
pub fn ramp_stop_count(node: &ShaderNode) -> usize {
    let n = node.value.round() as i32;
    if n >= 2 && n <= COLOR_RAMP_MAX_STOPS as i32 {
        n as usize
    } else {
        2
    }
}

pub fn ramp_get_stops(node: &ShaderNode) -> (Vec<[f32; 4]>, Vec<f32>, u8, u8) {
    let n = ramp_stop_count(node);
    let mut cols = Vec::with_capacity(n);
    let mut poss = Vec::with_capacity(n);
    cols.push(node.color);
    poss.push(node.props[12].clamp(0.0, 1.0));
    if n >= 2 {
        cols.push([
            node.props[0],
            node.props[1],
            node.props[2],
            if node.props[3].abs() < 1e-8 {
                1.0
            } else {
                node.props[3]
            },
        ]);
        // default white at 1 if uninitialized
        if node.props[0].abs() < 1e-8
            && node.props[1].abs() < 1e-8
            && node.props[2].abs() < 1e-8
            && node.props[3].abs() < 1e-8
            && node.props[13].abs() < 1e-8
            && node.props[12].abs() < 1e-8
        {
            cols[1] = [1.0, 1.0, 1.0, 1.0];
            poss[0] = 0.0;
            poss.push(1.0);
        } else {
            poss.push(node.props[13].clamp(0.0, 1.0));
        }
    }
    if n >= 3 {
        cols.push([
            node.props[4],
            node.props[5],
            node.props[6],
            if node.props[7].abs() < 1e-8 {
                1.0
            } else {
                node.props[7]
            },
        ]);
        poss.push(node.props[14].clamp(0.0, 1.0));
    }
    if n >= 4 {
        cols.push([
            node.props[8],
            node.props[9],
            node.props[10],
            if node.props[11].abs() < 1e-8 {
                1.0
            } else {
                node.props[11]
            },
        ]);
        poss.push(node.props[15].clamp(0.0, 1.0));
    }
    // Ensure monotonic-ish defaults for extra stops
    for i in 0..poss.len() {
        if i > 0 && poss[i] < poss[i - 1] && node.props[12 + i].abs() < 1e-8 {
            poss[i] = (i as f32) / ((n - 1).max(1) as f32);
        }
    }
    let interp = node.mode & 0x0f;
    let color_mode = (node.mode >> 4) & 0x03;
    (cols, poss, interp, color_mode)
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

fn mix_color_mode(a: [f32; 4], b: [f32; 4], factor: f32, mode: u8) -> [f32; 4] {
    let blend = |x: f32, y: f32| match mode {
        1 => x.min(y),
        2 => x * y,
        3 => x.max(y),
        4 => x + y - x * y,
        5 => {
            if x <= 0.5 {
                2.0 * x * y
            } else {
                1.0 - 2.0 * (1.0 - x) * (1.0 - y)
            }
        }
        6 => {
            if y <= 0.5 {
                x - (1.0 - 2.0 * y) * x * (1.0 - x)
            } else {
                x + (2.0 * y - 1.0) * (x.sqrt() - x)
            }
        }
        7 => (x + 2.0 * y - 1.0).clamp(0.0, 1.0),
        8 => (x - y).abs(),
        9 => x + y - 2.0 * x * y,
        10 => x + y,
        11 => x - y,
        12 => {
            if y.abs() > 1e-8 {
                x / y
            } else {
                0.0
            }
        }
        _ => y,
    };
    let mut out = [0.0; 4];
    for channel in 0..3 {
        let value = blend(a[channel], b[channel]);
        out[channel] = a[channel] + (value - a[channel]) * factor.clamp(0.0, 1.0);
    }
    out[3] = a[3] + (b[3] - a[3]) * factor.clamp(0.0, 1.0);
    out
}

/// Evaluate Blender-like Color Ramp at factor t in [0,1].
pub fn color_ramp_eval_stops(t: f32, cols: &[[f32; 4]], poss: &[f32], interp: u8) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    if cols.is_empty() {
        return [0.0, 0.0, 0.0, 1.0];
    }
    if cols.len() == 1 {
        return cols[0];
    }
    // Sort by position
    let mut order: Vec<usize> = (0..cols.len()).collect();
    order.sort_by(|&i, &j| {
        poss[i]
            .partial_cmp(&poss[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let first = order[0];
    let last = *order.last().unwrap();
    if t <= poss[first] {
        return cols[first];
    }
    if t >= poss[last] {
        return cols[last];
    }
    // Find segment
    for w in order.windows(2) {
        let i = w[0];
        let j = w[1];
        let a = poss[i];
        let b = poss[j];
        if t >= a && t <= b {
            let denom = (b - a).max(1e-6);
            let mut f = (t - a) / denom;
            f = match interp {
                1 => f * f * (3.0 - 2.0 * f), // Ease (smoothstep)
                2 => 0.0,                     // Constant: hold left color of this segment
                3 | 4 => {
                    // Cardinal / B-Spline: softer cubic-ish
                    f * f * (3.0 - 2.0 * f)
                }
                _ => f, // Linear
            };
            return mix4(cols[i], cols[j], f);
        }
    }
    cols[last]
}

pub fn color_ramp_from_node(node: &ShaderNode, fac: f32) -> [f32; 4] {
    let (cols, poss, interp, _cm) = ramp_get_stops(node);
    color_ramp_eval_stops(fac, &cols, &poss, interp)
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

fn socket_default_val(node: &ShaderNode, socket: u8) -> Option<SocketVal> {
    let def = node.kind.inputs().get(socket as usize)?;
    let value = node.input_defaults.get(socket as usize)?;
    Some(match def.ty {
        SocketType::Float => SocketVal::Scalar(value[0]),
        SocketType::Color | SocketType::Shader => SocketVal::Color(*value),
        SocketType::Vector => SocketVal::Vector([value[0], value[1], value[2]]),
        SocketType::String => return None,
    })
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

/// `live`: skip generating new procedural textures (use mid gray / existing image indices only).
/// `frame` / `time_sec`: drive Scene Time and similar inputs.
pub fn evaluate_graph(
    graph: &ShaderGraph,
    material: &Material,
    textures: &[TextureData],
    tex_size: u32,
) -> GraphEvalResult {
    evaluate_graph_ex(graph, material, textures, tex_size, false, 1.0, 0.0)
}

fn evaluate_graph_center(
    graph: &ShaderGraph,
    material: &Material,
    textures: &[TextureData],
    tex_size: u32,
    live: bool,
    frame: f32,
    time_sec: f32,
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
        displacement_scale: material.displacement_scale,
        volume_density: material.volume_density,
        ior: 1.45,
        alpha: material.base_color[3],
        emission_strength: 1.0,
    };

    let order = topo_order(graph);
    let node_map: HashMap<u64, &ShaderNode> = graph.nodes.iter().map(|n| (n.id, n)).collect();
    let mut cache: HashMap<(u64, u8), SocketVal> = HashMap::new();
    let mut proc_tex_index: HashMap<u64, usize> = HashMap::new();

    for id in order {
        let Some(node) = node_map.get(&id) else {
            continue;
        };
        let ins = node.kind.inputs();
        let outs = node.kind.outputs();
        // Helper: linked input or default from node.value/color/props
        let inp = |sock: u8, default: SocketVal| -> SocketVal {
            find_input(graph, id, sock, &cache)
                .or_else(|| socket_default_val(node, sock))
                .unwrap_or(default)
        };

        match node.kind {
            ShaderNodeKind::Value => {
                cache.insert((id, 0), SocketVal::Scalar(node.value));
            }
            ShaderNodeKind::Rgb => {
                cache.insert((id, 0), SocketVal::Color(node.color));
                cache.insert((id, 1), SocketVal::Scalar(node.color[3]));
            }
            ShaderNodeKind::TexCoord | ShaderNodeKind::UvMap => {
                cache.insert((id, 0), SocketVal::Vector([0.5, 0.5, 0.0]));
            }
            ShaderNodeKind::Mapping => {
                let v = inp(0, SocketVal::Vector([0.5, 0.5, 0.0])).as_vector();
                let loc = inp(1, SocketVal::Vector([0.0, 0.0, 0.0])).as_vector();
                let rot = inp(2, SocketVal::Vector([0.0, 0.0, 0.0])).as_vector();
                let scale = inp(
                    3,
                    SocketVal::Vector([
                        if node.value.abs() < 1e-6 {
                            1.0
                        } else {
                            node.value
                        },
                        if node.props[0].abs() < 1e-6 {
                            1.0
                        } else {
                            node.props[0]
                        },
                        if node.props[1].abs() < 1e-6 {
                            1.0
                        } else {
                            node.props[1]
                        },
                    ]),
                )
                .as_vector();
                let p = rotate_euler_xyz([v[0] * scale[0], v[1] * scale[1], v[2] * scale[2]], rot);
                cache.insert(
                    (id, 0),
                    SocketVal::Vector([p[0] + loc[0], p[1] + loc[1], p[2] + loc[2]]),
                );
            }
            ShaderNodeKind::ImageTexture => {
                // mode: interp in low 2 bits; projection <<2; extension <<4
                if let Some(ti) = node.image_index.filter(|i| *i < textures.len()) {
                    let uv = inp(0, SocketVal::Vector([0.5, 0.5, 0.0])).as_vector();
                    let mut u = uv[0];
                    let mut v = uv[1];
                    let ext = (node.mode / 16) % 4;
                    match ext {
                        1 => {
                            u = u.clamp(0.0, 1.0);
                            v = v.clamp(0.0, 1.0);
                        }
                        2 => {
                            if u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 {
                                cache.insert((id, 0), SocketVal::Color([0.0, 0.0, 0.0, 0.0]));
                                cache.insert((id, 1), SocketVal::Scalar(0.0));
                                proc_tex_index.insert(id, ti);
                                continue;
                            }
                        }
                        3 => {
                            u = u.abs() % 2.0;
                            if u > 1.0 {
                                u = 2.0 - u;
                            }
                            v = v.abs() % 2.0;
                            if v > 1.0 {
                                v = 2.0 - v;
                            }
                        }
                        _ => {
                            u = u.rem_euclid(1.0);
                            v = v.rem_euclid(1.0);
                        }
                    }
                    let c = sample_tex(&textures[ti], u, v);
                    cache.insert((id, 0), SocketVal::Color(c));
                    cache.insert((id, 1), SocketVal::Scalar(c[3]));
                    proc_tex_index.insert(id, ti);
                } else {
                    cache.insert((id, 0), SocketVal::Color([1.0, 1.0, 1.0, 1.0]));
                    cache.insert((id, 1), SocketVal::Scalar(1.0));
                }
            }
            ShaderNodeKind::Noise => {
                if live {
                    let (col, fac) = proc_sample(
                        node.kind,
                        node.mode,
                        tex_scale(node),
                        &node.props,
                        node.color,
                        0.37,
                        0.61,
                    );
                    cache.insert((id, 0), SocketVal::Scalar(fac));
                    cache.insert((id, 1), SocketVal::Color(col));
                    continue;
                }
                let scale = {
                    let s = inp(2, SocketVal::Scalar(tex_scale(node))).as_scalar();
                    if s.abs() < 1e-4 {
                        5.0
                    } else {
                        s
                    }
                };
                let detail = inp(
                    3,
                    SocketVal::Scalar(if node.props[0] > 0.0 {
                        node.props[0]
                    } else {
                        5.0
                    }),
                )
                .as_scalar()
                .clamp(0.0, 15.0);
                let roughness = inp(
                    4,
                    SocketVal::Scalar(if node.props[1] > 0.0 {
                        node.props[1]
                    } else {
                        0.5
                    }),
                )
                .as_scalar()
                .clamp(0.0, 1.0);
                let lacunarity = inp(
                    5,
                    SocketVal::Scalar(if node.props[2] > 0.0 {
                        node.props[2]
                    } else {
                        2.0
                    }),
                )
                .as_scalar()
                .max(1.0);
                let distortion = inp(6, SocketVal::Scalar(node.props[3])).as_scalar();
                let size = tex_size.max(64).min(512);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32;
                        let v = y as f32 / size as f32;
                        let mut px = u * scale;
                        let mut py = v * scale;
                        if distortion.abs() > 1e-6 {
                            px += distortion * noise2(px + 17.1, py + 9.2);
                            py += distortion * noise2(px + 3.7, py + 28.4);
                        }
                        // Multifractal-ish fbm with roughness/lacunarity
                        let mut amp = 1.0;
                        let mut freq = 1.0;
                        let mut sum = 0.0;
                        let mut max_amp = 0.0;
                        let oct = detail.floor() as i32 + 1;
                        for _ in 0..oct.max(1).min(12) {
                            sum += amp * noise2(px * freq, py * freq);
                            max_amp += amp;
                            amp *= roughness;
                            freq *= lacunarity;
                        }
                        let n = if max_amp > 1e-6 {
                            (sum / max_amp) * 0.5 + 0.5
                        } else {
                            0.5
                        };
                        let i = ((y * size + x) * 4) as usize;
                        let b = (n.clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i] = b;
                        rgba[i + 1] = b;
                        rgba[i + 2] = b;
                        rgba[i + 3] = 255;
                    }
                }
                let tex = TextureData {
                    name: format!("__sg_Noise_{id}"),
                    width: size,
                    height: size,
                    rgba8: rgba,
                };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                let fac = (mid[0] + mid[1] + mid[2]) / 3.0;
                cache.insert((id, 0), SocketVal::Scalar(fac));
                cache.insert((id, 1), SocketVal::Color(mid));
            }
            ShaderNodeKind::Voronoi => {
                if live {
                    let (col, fac) = proc_sample(
                        node.kind,
                        node.mode,
                        tex_scale(node),
                        &node.props,
                        node.color,
                        0.37,
                        0.61,
                    );
                    cache.insert((id, 0), SocketVal::Scalar(fac));
                    cache.insert((id, 1), SocketVal::Color(col));
                    cache.insert((id, 2), SocketVal::Vector([0.37, 0.61, 0.0]));
                    cache.insert((id, 3), SocketVal::Scalar(0.0));
                    cache.insert((id, 4), SocketVal::Scalar(fac));
                    continue;
                }
                let scale = {
                    let s = inp(2, SocketVal::Scalar(tex_scale(node))).as_scalar();
                    if s.abs() < 1e-4 {
                        5.0
                    } else {
                        s
                    }
                };
                let randomness = inp(
                    7,
                    SocketVal::Scalar(if node.props[0] > 0.0 {
                        node.props[0]
                    } else {
                        1.0
                    }),
                )
                .as_scalar()
                .clamp(0.0, 1.0);
                let smoothness = inp(6, SocketVal::Scalar(0.5)).as_scalar();
                let size = tex_size.max(64).min(512);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                let mut mid_d = 0.5f32;
                let mut mid_pos = [0.5f32, 0.5, 0.0];
                let mut mid_radius = 0.0f32;
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32;
                        let v = y as f32 / size as f32;
                        let (d, cell_col, position, radius) = voronoi_sample(
                            u * scale,
                            v * scale,
                            randomness,
                            smoothness,
                            node.mode,
                        );
                        let i = ((y * size + x) * 4) as usize;
                        let b = (d.clamp(0.0, 1.0) * 255.0) as u8;
                        let c0 = (cell_col[0] * 255.0) as u8;
                        let c1 = (cell_col[1] * 255.0) as u8;
                        let c2 = (cell_col[2] * 255.0) as u8;
                        // Store distance as grayscale for Fac; color from cell id
                        rgba[i] = c0;
                        rgba[i + 1] = c1;
                        rgba[i + 2] = c2;
                        rgba[i + 3] = b;
                        if x == size / 2 && y == size / 2 {
                            mid_d = d;
                            mid_pos = position;
                            mid_radius = radius;
                        }
                    }
                }
                let tex = TextureData {
                    name: format!("__sg_Voronoi_{id}"),
                    width: size,
                    height: size,
                    rgba8: rgba,
                };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Scalar(mid_d));
                cache.insert((id, 1), SocketVal::Color([mid[0], mid[1], mid[2], 1.0]));
                cache.insert((id, 2), SocketVal::Vector(mid_pos));
                cache.insert((id, 3), SocketVal::Scalar(mid_pos[2]));
                cache.insert((id, 4), SocketVal::Scalar(mid_radius));
            }
            ShaderNodeKind::Brick => {
                if live {
                    let (col, fac) = proc_sample(
                        node.kind,
                        node.mode,
                        tex_scale(node),
                        &node.props,
                        node.color,
                        0.37,
                        0.61,
                    );
                    cache.insert((id, 0), SocketVal::Color(col));
                    cache.insert((id, 1), SocketVal::Scalar(fac));
                    continue;
                }
                let c1 = inp(1, SocketVal::Color([0.8, 0.8, 0.8, 1.0])).as_color();
                let c2 = inp(2, SocketVal::Color([0.2, 0.2, 0.2, 1.0])).as_color();
                let mortar = inp(3, SocketVal::Color([0.0, 0.0, 0.0, 1.0])).as_color();
                let scale = inp(4, SocketVal::Scalar(tex_scale(node))).as_scalar();
                let mortar_size = inp(5, SocketVal::Scalar(0.02)).as_scalar();
                let mortar_smooth = inp(6, SocketVal::Scalar(0.1)).as_scalar();
                let bias = inp(7, SocketVal::Scalar(0.0)).as_scalar();
                let brick_w = inp(8, SocketVal::Scalar(0.5)).as_scalar();
                let row_h = inp(9, SocketVal::Scalar(0.25)).as_scalar();
                let (offset, offset_frequency, squash, squash_frequency) = brick_controls(node);
                let size = tex_size.max(64).min(512);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let point = [x as f32 / size as f32, y as f32 / size as f32, 0.0];
                        let (col, fac) = brick_texture(
                            point,
                            scale,
                            c1,
                            c2,
                            mortar,
                            mortar_size,
                            mortar_smooth,
                            bias,
                            brick_w,
                            row_h,
                            offset,
                            offset_frequency,
                            squash,
                            squash_frequency,
                        );
                        let i = ((y * size + x) * 4) as usize;
                        rgba[i] = (col[0].clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 1] = (col[1].clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 2] = (col[2].clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 3] = (fac.clamp(0.0, 1.0) * 255.0) as u8;
                    }
                }
                let tex = TextureData {
                    name: format!("__sg_Brick_{id}"),
                    width: size,
                    height: size,
                    rgba8: rgba,
                };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Color([mid[0], mid[1], mid[2], 1.0]));
                cache.insert((id, 1), SocketVal::Scalar(mid[3]));
            }
            ShaderNodeKind::Checker => {
                if live {
                    let (col, fac) = proc_sample(
                        node.kind,
                        node.mode,
                        tex_scale(node),
                        &node.props,
                        node.color,
                        0.37,
                        0.61,
                    );
                    cache.insert((id, 0), SocketVal::Color(col));
                    cache.insert((id, 1), SocketVal::Scalar(fac));
                    continue;
                }
                let c1 = inp(1, SocketVal::Color([1.0, 1.0, 1.0, 1.0])).as_color();
                let c2 = inp(2, SocketVal::Color([0.0, 0.0, 0.0, 1.0])).as_color();
                let scale = inp(3, SocketVal::Scalar(tex_scale(node)))
                    .as_scalar()
                    .max(0.01);
                let size = tex_size.max(64).min(512);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32;
                        let v = y as f32 / size as f32;
                        let cx = (u * scale).floor() as i32;
                        let cy = (v * scale).floor() as i32;
                        let fac = if (cx + cy) % 2 == 0 { 1.0 } else { 0.0 };
                        let col = if fac > 0.5 { c1 } else { c2 };
                        let i = ((y * size + x) * 4) as usize;
                        rgba[i] = (col[0].clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 1] = (col[1].clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 2] = (col[2].clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 3] = 255;
                    }
                }
                let tex = TextureData {
                    name: format!("__sg_Checker_{id}"),
                    width: size,
                    height: size,
                    rgba8: rgba,
                };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Color(mid));
                cache.insert((id, 1), SocketVal::Scalar(1.0));
            }
            ShaderNodeKind::Gradient => {
                // mode: 0 Linear, 1 Quadratic, 2 Easing, 3 Diagonal, 4 Spherical, 5 Radial
                if live {
                    let (col, fac) = proc_sample(
                        node.kind,
                        node.mode,
                        tex_scale(node),
                        &node.props,
                        node.color,
                        0.37,
                        0.61,
                    );
                    cache.insert((id, 0), SocketVal::Scalar(fac));
                    cache.insert((id, 1), SocketVal::Color(col));
                    continue;
                }
                let size = tex_size.max(64).min(512);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32;
                        let v = y as f32 / size as f32;
                        let t = match node.mode {
                            1 => u * u, // quadratic
                            2 => {
                                // easing
                                let x = u.clamp(0.0, 1.0);
                                x * x * (3.0 - 2.0 * x)
                            }
                            3 => ((u + v) * 0.5).clamp(0.0, 1.0), // diagonal
                            4 => {
                                let dx = u - 0.5;
                                let dy = v - 0.5;
                                (1.0 - (dx * dx + dy * dy).sqrt() * 2.0).clamp(0.0, 1.0)
                            }
                            5 => {
                                // radial
                                let a = (v - 0.5).atan2(u - 0.5);
                                (a / (2.0 * std::f32::consts::PI) + 0.5).fract()
                            }
                            _ => u.clamp(0.0, 1.0), // linear
                        };
                        let i = ((y * size + x) * 4) as usize;
                        let b = (t * 255.0) as u8;
                        rgba[i] = b;
                        rgba[i + 1] = b;
                        rgba[i + 2] = b;
                        rgba[i + 3] = 255;
                    }
                }
                let tex = TextureData {
                    name: format!("__sg_Gradient_{id}"),
                    width: size,
                    height: size,
                    rgba8: rgba,
                };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                let fac = mid[0];
                cache.insert((id, 0), SocketVal::Scalar(fac));
                cache.insert((id, 1), SocketVal::Color(mid));
            }
            ShaderNodeKind::Wave => {
                if live {
                    let (col, fac) = proc_sample(
                        node.kind,
                        node.mode,
                        tex_scale(node),
                        &node.props,
                        node.color,
                        0.37,
                        0.61,
                    );
                    cache.insert((id, 0), SocketVal::Scalar(fac));
                    cache.insert((id, 1), SocketVal::Color(col));
                    continue;
                }
                let scale = inp(1, SocketVal::Scalar(tex_scale(node))).as_scalar();
                let distortion = inp(2, SocketVal::Scalar(0.0)).as_scalar();
                let detail = inp(3, SocketVal::Scalar(2.0)).as_scalar();
                let detail_scale = inp(4, SocketVal::Scalar(1.0)).as_scalar();
                let detail_roughness = inp(5, SocketVal::Scalar(0.5)).as_scalar();
                let phase = inp(6, SocketVal::Scalar(0.0)).as_scalar();
                let size = tex_size.max(64).min(512);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32;
                        let v = y as f32 / size as f32;
                        let t = wave_value(
                            [u, v, 0.0],
                            scale,
                            distortion,
                            detail,
                            detail_scale,
                            detail_roughness,
                            phase,
                            node.mode,
                            node.props[8].round().clamp(0.0, 3.0) as u8,
                            node.props[9].round().clamp(0.0, 3.0) as u8,
                            node.props[10].round().clamp(0.0, 2.0) as u8,
                        );
                        let i = ((y * size + x) * 4) as usize;
                        let b = (t.clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i] = b;
                        rgba[i + 1] = b;
                        rgba[i + 2] = b;
                        rgba[i + 3] = 255;
                    }
                }
                let tex = TextureData {
                    name: format!("__sg_Wave_{id}"),
                    width: size,
                    height: size,
                    rgba8: rgba,
                };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Scalar(mid[0]));
                cache.insert((id, 1), SocketVal::Color(mid));
            }
            ShaderNodeKind::Magic => {
                if live {
                    let (col, fac) = proc_sample(
                        node.kind,
                        node.mode,
                        tex_scale(node),
                        &node.props,
                        node.color,
                        0.37,
                        0.61,
                    );
                    cache.insert((id, 0), SocketVal::Scalar(fac));
                    cache.insert((id, 1), SocketVal::Color(col));
                    continue;
                }
                let scale = inp(1, SocketVal::Scalar(tex_scale(node))).as_scalar();
                let distortion = inp(2, SocketVal::Scalar(1.0)).as_scalar();
                let depth = magic_depth(node);
                let size = tex_size.max(64).min(512);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let point = [x as f32 / size as f32, y as f32 / size as f32, 0.0];
                        let (color, _) = magic_texture(point, scale, distortion, depth);
                        let i = ((y * size + x) * 4) as usize;
                        rgba[i] = (color[0].clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 1] = (color[1].clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 2] = (color[2].clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 3] = 255;
                    }
                }
                let tex = TextureData {
                    name: format!("__sg_Magic_{id}"),
                    width: size,
                    height: size,
                    rgba8: rgba,
                };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Scalar((mid[0] + mid[1] + mid[2]) / 3.0));
                cache.insert((id, 1), SocketVal::Color(mid));
            }
            ShaderNodeKind::Gabor => {
                if live {
                    let (col, fac) = proc_sample(
                        node.kind,
                        node.mode,
                        tex_scale(node),
                        &node.props,
                        node.color,
                        0.37,
                        0.61,
                    );
                    cache.insert((id, 0), SocketVal::Scalar(fac));
                    cache.insert((id, 1), SocketVal::Color(col));
                    continue;
                }
                let scale = inp(1, SocketVal::Scalar(5.0)).as_scalar().max(0.01);
                let frequency = inp(
                    2,
                    SocketVal::Scalar(if node.props[0] > 0.0 {
                        node.props[0]
                    } else {
                        2.0
                    }),
                )
                .as_scalar();
                let aniso = inp(3, SocketVal::Scalar(1.0)).as_scalar().max(0.01);
                let orient = inp(4, SocketVal::Scalar(0.0)).as_scalar();
                let size = tex_size.max(64).min(512);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                let ca = orient.cos();
                let sa = orient.sin();
                for y in 0..size {
                    for x in 0..size {
                        let u = (x as f32 / size as f32 - 0.5) * scale;
                        let v = (y as f32 / size as f32 - 0.5) * scale;
                        let rx = u * ca + v * sa;
                        let ry = (-u * sa + v * ca) * aniso;
                        let gauss = (-(rx * rx + ry * ry)).exp();
                        let wave = (rx * frequency * std::f32::consts::TAU).cos();
                        let t = (gauss * wave * 0.5 + 0.5).clamp(0.0, 1.0);
                        let i = ((y * size + x) * 4) as usize;
                        let b = (t * 255.0) as u8;
                        rgba[i] = b;
                        rgba[i + 1] = b;
                        rgba[i + 2] = b;
                        rgba[i + 3] = 255;
                    }
                }
                let tex = TextureData {
                    name: format!("__sg_Gabor_{id}"),
                    width: size,
                    height: size,
                    rgba8: rgba,
                };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Scalar(mid[0]));
                cache.insert((id, 1), SocketVal::Color(mid));
            }
            ShaderNodeKind::WhiteNoise => {
                if live {
                    let (col, fac) = proc_sample(
                        node.kind,
                        node.mode,
                        tex_scale(node),
                        &node.props,
                        node.color,
                        0.37,
                        0.61,
                    );
                    cache.insert((id, 0), SocketVal::Scalar(fac));
                    cache.insert((id, 1), SocketVal::Color(col));
                    continue;
                }
                let size = tex_size.max(64).min(512);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                let w = inp(1, SocketVal::Scalar(0.0)).as_scalar();
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32;
                        let v = y as f32 / size as f32;
                        let (_, color) = white_noise_texture([u, v, 0.0], w, node.mode);
                        let i = ((y * size + x) * 4) as usize;
                        rgba[i] = (color[0] * 255.0) as u8;
                        rgba[i + 1] = (color[1] * 255.0) as u8;
                        rgba[i + 2] = (color[2] * 255.0) as u8;
                        rgba[i + 3] = 255;
                    }
                }
                let tex = TextureData {
                    name: format!("__sg_WhiteNoise_{id}"),
                    width: size,
                    height: size,
                    rgba8: rgba,
                };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Scalar(mid[0]));
                cache.insert((id, 1), SocketVal::Color(mid));
            }
            ShaderNodeKind::Sky => {
                // Approximate Nishita-like: blue zenith, warmer horizon; mode unused
                let size = if live { 64 } else { tex_size.max(64).min(256) };
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let v = 1.0 - y as f32 / size as f32; // up
                        let t = v.clamp(0.0, 1.0);
                        let r = 0.45 + 0.35 * (1.0 - t);
                        let g = 0.55 + 0.25 * (1.0 - t) + 0.15 * t;
                        let b = 0.75 + 0.2 * t;
                        let i = ((y * size + x) * 4) as usize;
                        rgba[i] = (r.clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i + 3] = 255;
                    }
                }
                if !live {
                    let tex = TextureData {
                        name: format!("__sg_Sky_{id}"),
                        width: size,
                        height: size,
                        rgba8: rgba.clone(),
                    };
                    let new_idx = textures.len() + result.new_textures.len();
                    result.new_textures.push(tex);
                    proc_tex_index.insert(id, new_idx);
                }
                cache.insert((id, 0), SocketVal::Color([0.55, 0.65, 0.9, 1.0]));
            }
            ShaderNodeKind::Ies => {
                // Radial falloff as stand-in for IES profile (no .ies file load)
                let strength = inp(
                    1,
                    SocketVal::Scalar(if node.value.abs() < 1e-6 {
                        1.0
                    } else {
                        node.value
                    }),
                )
                .as_scalar();
                if live {
                    let (col, fac) = proc_sample(
                        node.kind,
                        node.mode,
                        strength,
                        &node.props,
                        node.color,
                        0.37,
                        0.61,
                    );
                    cache.insert((id, 0), SocketVal::Scalar(fac * strength));
                    cache.insert(
                        (id, 1),
                        SocketVal::Color([
                            col[0] * strength,
                            col[1] * strength,
                            col[2] * strength,
                            1.0,
                        ]),
                    );
                    continue;
                }
                let size = tex_size.max(64).min(256);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32 - 0.5;
                        let v = y as f32 / size as f32 - 0.5;
                        let d = (u * u + v * v).sqrt() * 2.0;
                        let t = ((1.0 - d).max(0.0).powf(2.0) * strength).clamp(0.0, 1.0);
                        let i = ((y * size + x) * 4) as usize;
                        let b = (t * 255.0) as u8;
                        rgba[i] = b;
                        rgba[i + 1] = b;
                        rgba[i + 2] = b;
                        rgba[i + 3] = 255;
                    }
                }
                let tex = TextureData {
                    name: format!("IES_{id}"),
                    width: size,
                    height: size,
                    rgba8: rgba,
                };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Scalar(mid[0]));
                cache.insert((id, 1), SocketVal::Color(mid));
            }
            ShaderNodeKind::Environment => {
                // Use linked image if present, else neutral
                if let Some(ti) = node.image_index.filter(|i| *i < textures.len()) {
                    let c = sample_tex(&textures[ti], 0.5, 0.5);
                    cache.insert((id, 0), SocketVal::Color(c));
                    cache.insert((id, 1), SocketVal::Scalar(c[3]));
                    proc_tex_index.insert(id, ti);
                } else {
                    cache.insert((id, 0), SocketVal::Color([0.5, 0.55, 0.65, 1.0]));
                    cache.insert((id, 1), SocketVal::Scalar(1.0));
                }
            }
            ShaderNodeKind::ColorRamp => {
                // Fac → Color + Alpha (Blender manual)
                let fac = inp(0, SocketVal::Scalar(0.5)).as_scalar();
                let col = color_ramp_from_node(node, fac);
                cache.insert((id, 0), SocketVal::Color(col));
                cache.insert((id, 1), SocketVal::Scalar(col[3])); // Alpha
            }
            ShaderNodeKind::Mix => {
                let fac = inp(0, SocketVal::Scalar(0.5)).as_scalar();
                let a = inp(1, SocketVal::Color([0.0, 0.0, 0.0, 1.0])).as_color();
                let b = inp(2, SocketVal::Color([1.0, 1.0, 1.0, 1.0])).as_color();
                cache.insert(
                    (id, 0),
                    SocketVal::Color(mix_color_mode(a, b, fac, node.mode)),
                );
            }
            ShaderNodeKind::Blackbody => {
                let t = inp(
                    0,
                    SocketVal::Scalar(if node.value > 100.0 {
                        node.value
                    } else {
                        6500.0
                    }),
                )
                .as_scalar();
                // Rough Planck approximation to RGB
                let x = (t / 1000.0).clamp(1.0, 40.0);
                let r = if x <= 6.6 {
                    1.0
                } else {
                    (1.292_936 * (x - 6.0).powf(-0.133)).clamp(0.0, 1.0)
                };
                let g = if x <= 6.6 {
                    (0.390_081 * x.ln() - 0.631).clamp(0.0, 1.0)
                } else {
                    (1.196_040 * (x - 6.0).powf(-0.075)).clamp(0.0, 1.0)
                };
                let b = if x >= 6.6 {
                    1.0
                } else if x <= 1.9 {
                    0.0
                } else {
                    (0.543_206 * (x - 1.0).ln() - 0.370).clamp(0.0, 1.0)
                };
                cache.insert((id, 0), SocketVal::Color([r, g, b, 1.0]));
            }
            ShaderNodeKind::Wavelength => {
                let nm = inp(
                    0,
                    SocketVal::Scalar(if node.value > 10.0 { node.value } else { 550.0 }),
                )
                .as_scalar();
                // Simple rainbow map 380-780
                let t = ((nm - 380.0) / 400.0).clamp(0.0, 1.0);
                let r = (1.0 - (t - 0.0).abs() * 3.0).clamp(0.0, 1.0);
                let g = (1.0 - (t - 0.4).abs() * 3.0).clamp(0.0, 1.0);
                let b = (1.0 - (t - 0.8).abs() * 3.0).clamp(0.0, 1.0);
                cache.insert((id, 0), SocketVal::Color([r, g, b, 1.0]));
            }
            ShaderNodeKind::BrightContrast => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                let bright = inp(1, SocketVal::Scalar(node.props[0])).as_scalar();
                let contrast = inp(
                    2,
                    SocketVal::Scalar(if node.props[1].abs() < 1e-6 {
                        0.0
                    } else {
                        node.props[1]
                    }),
                )
                .as_scalar();
                let f = |x: f32| ((x - 0.5) * (1.0 + contrast) + 0.5 + bright).clamp(0.0, 1.0);
                cache.insert((id, 0), SocketVal::Color([f(c[0]), f(c[1]), f(c[2]), c[3]]));
            }
            ShaderNodeKind::Gamma => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                let g = inp(
                    1,
                    SocketVal::Scalar(if node.value.abs() < 1e-6 {
                        1.0
                    } else {
                        node.value
                    }),
                )
                .as_scalar()
                .max(0.01);
                cache.insert(
                    (id, 0),
                    SocketVal::Color([
                        c[0].max(0.0).powf(1.0 / g),
                        c[1].max(0.0).powf(1.0 / g),
                        c[2].max(0.0).powf(1.0 / g),
                        c[3],
                    ]),
                );
            }
            ShaderNodeKind::InvertColor => {
                let fac = inp(0, SocketVal::Scalar(1.0)).as_scalar();
                let c = inp(1, SocketVal::Color(node.color)).as_color();
                let inv = [1.0 - c[0], 1.0 - c[1], 1.0 - c[2], c[3]];
                cache.insert((id, 0), SocketVal::Color(mix4(c, inv, fac)));
            }
            ShaderNodeKind::HueSatVal => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                let fac = inp(4, SocketVal::Scalar(1.0)).as_scalar();
                // Pass-through with value scale as approximate
                let val = inp(3, SocketVal::Scalar(1.0)).as_scalar();
                let sat = inp(2, SocketVal::Scalar(1.0)).as_scalar();
                let avg = (c[0] + c[1] + c[2]) / 3.0;
                let mut o = [
                    (avg + (c[0] - avg) * sat) * val,
                    (avg + (c[1] - avg) * sat) * val,
                    (avg + (c[2] - avg) * sat) * val,
                    c[3],
                ];
                o = mix4(c, o, fac);
                cache.insert((id, 0), SocketVal::Color(o));
            }
            ShaderNodeKind::RgbToBw => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                let y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                cache.insert((id, 0), SocketVal::Scalar(y));
            }
            ShaderNodeKind::CombineColor | ShaderNodeKind::CombineXyz => {
                let x = inp(0, SocketVal::Scalar(0.0)).as_scalar();
                let y = inp(1, SocketVal::Scalar(0.0)).as_scalar();
                let z = inp(2, SocketVal::Scalar(0.0)).as_scalar();
                cache.insert((id, 0), SocketVal::Color([x, y, z, 1.0]));
            }
            ShaderNodeKind::SeparateColor | ShaderNodeKind::SeparateXyz => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                cache.insert((id, 0), SocketVal::Scalar(c[0]));
                cache.insert((id, 1), SocketVal::Scalar(c[1]));
                cache.insert((id, 2), SocketVal::Scalar(c[2]));
            }
            ShaderNodeKind::Math => {
                let a = inp(0, SocketVal::Scalar(node.value)).as_scalar();
                let b = inp(1, SocketVal::Scalar(node.props[0])).as_scalar();
                let r = math_op(node.mode, a, b);
                cache.insert((id, 0), SocketVal::Scalar(r));
            }
            ShaderNodeKind::Script => {
                // Simple expression on a,b → result. name_prop holds expression like "a*b+0.5"
                let a = inp(0, SocketVal::Scalar(node.value)).as_scalar();
                let b = inp(1, SocketVal::Scalar(node.props[0])).as_scalar();
                let expr = if node.name_prop.is_empty() {
                    "a"
                } else {
                    node.name_prop.as_str()
                };
                let r = eval_simple_expr(expr, a, b);
                cache.insert((id, 0), SocketVal::Color([r, r, r, 1.0]));
            }
            ShaderNodeKind::MapRange => {
                let v = inp(0, SocketVal::Scalar(node.value)).as_scalar();
                let fmin = inp(1, SocketVal::Scalar(0.0)).as_scalar();
                let fmax = inp(2, SocketVal::Scalar(1.0)).as_scalar();
                let tmin = inp(3, SocketVal::Scalar(0.0)).as_scalar();
                let tmax = inp(4, SocketVal::Scalar(1.0)).as_scalar();
                let t = if (fmax - fmin).abs() < 1e-8 {
                    0.0
                } else {
                    (v - fmin) / (fmax - fmin)
                };
                let t = if node.mode == 0 { t.clamp(0.0, 1.0) } else { t };
                cache.insert((id, 0), SocketVal::Scalar(tmin + t * (tmax - tmin)));
            }
            ShaderNodeKind::Clamp => {
                let v = inp(0, SocketVal::Scalar(node.value)).as_scalar();
                let mn = inp(1, SocketVal::Scalar(0.0)).as_scalar();
                let mx = inp(2, SocketVal::Scalar(1.0)).as_scalar();
                cache.insert((id, 0), SocketVal::Scalar(v.clamp(mn, mx)));
            }
            ShaderNodeKind::VectorMath => {
                let a = inp(0, SocketVal::Vector([0.0, 0.0, 0.0])).as_vector();
                let b = inp(1, SocketVal::Vector([0.0, 0.0, 0.0])).as_vector();
                let s = inp(2, SocketVal::Scalar(1.0)).as_scalar();
                let r = match node.mode {
                    1 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
                    2 => [a[0] * b[0], a[1] * b[1], a[2] * b[2]],
                    3 => [a[0] * s, a[1] * s, a[2] * s],
                    4 => {
                        let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-8);
                        [a[0] / l, a[1] / l, a[2] / l]
                    }
                    5 => {
                        // cross
                        [
                            a[1] * b[2] - a[2] * b[1],
                            a[2] * b[0] - a[0] * b[2],
                            a[0] * b[1] - a[1] * b[0],
                        ]
                    }
                    6 => {
                        let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                        [d, d, d]
                    }
                    7 => [a[0].abs(), a[1].abs(), a[2].abs()],
                    8 => [-a[0], -a[1], -a[2]],
                    9 => [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])],
                    10 => [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])],
                    11 => {
                        let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
                        [l, l, l]
                    }
                    _ => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
                };
                cache.insert((id, 0), SocketVal::Vector(r));
            }
            ShaderNodeKind::Bump => {
                let strength = inp(
                    0,
                    SocketVal::Scalar(if node.value.abs() < 1e-6 {
                        1.0
                    } else {
                        node.value
                    }),
                )
                .as_scalar();
                result.normal_scale = strength;
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 1.0]));
            }
            ShaderNodeKind::NormalMap => {
                let strength = inp(0, SocketVal::Scalar(1.0)).as_scalar();
                result.normal_scale = strength;
                if let Some(ti) = node.image_index.filter(|i| *i < textures.len()) {
                    result.normal_tex = Some(ti);
                }
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 1.0]));
            }
            ShaderNodeKind::Fresnel | ShaderNodeKind::LayerWeight => {
                let ior = inp(
                    0,
                    SocketVal::Scalar(if node.value.abs() < 1e-6 {
                        1.45
                    } else {
                        node.value
                    }),
                )
                .as_scalar();
                let f0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
                let f = f0 + (1.0 - f0) * 0.5;
                cache.insert((id, 0), SocketVal::Scalar(f.clamp(0.0, 1.0)));
                cache.insert((id, 1), SocketVal::Scalar((1.0 - f).clamp(0.0, 1.0)));
            }
            ShaderNodeKind::Emission | ShaderNodeKind::Background => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                let strength = inp(
                    1,
                    SocketVal::Scalar(if node.value.abs() < 1e-6 {
                        1.0
                    } else {
                        node.value
                    }),
                )
                .as_scalar();
                cache.insert(
                    (id, 0),
                    SocketVal::Color([c[0] * strength, c[1] * strength, c[2] * strength, c[3]]),
                );
            }
            ShaderNodeKind::Diffuse
            | ShaderNodeKind::Glossy
            | ShaderNodeKind::Transparent
            | ShaderNodeKind::Translucent
            | ShaderNodeKind::Sheen
            | ShaderNodeKind::Glass
            | ShaderNodeKind::Refraction
            | ShaderNodeKind::Toon
            | ShaderNodeKind::Subsurface
            | ShaderNodeKind::Metallic
            | ShaderNodeKind::Specular
            | ShaderNodeKind::Hair
            | ShaderNodeKind::Holdout => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                cache.insert((id, 0), SocketVal::Color(c));
            }
            ShaderNodeKind::AddShader | ShaderNodeKind::MixShader => {
                let a = inp(
                    if node.kind == ShaderNodeKind::MixShader {
                        1
                    } else {
                        0
                    },
                    SocketVal::Color([0.0, 0.0, 0.0, 1.0]),
                )
                .as_color();
                let b = inp(
                    if node.kind == ShaderNodeKind::MixShader {
                        2
                    } else {
                        1
                    },
                    SocketVal::Color([0.0, 0.0, 0.0, 1.0]),
                )
                .as_color();
                let fac = if node.kind == ShaderNodeKind::MixShader {
                    inp(0, SocketVal::Scalar(0.5)).as_scalar()
                } else {
                    0.5
                };
                cache.insert((id, 0), SocketVal::Color(mix4(a, b, fac)));
            }
            ShaderNodeKind::Principled => {
                let base_raw = inp(0, SocketVal::Color(node.color));
                let base = match base_raw {
                    SocketVal::Scalar(s) => [s, s, s, 1.0],
                    SocketVal::Color(c) => c,
                    SocketVal::Vector(v) => [v[0], v[1], v[2], 1.0],
                };
                let metallic = inp(1, SocketVal::Scalar(node.props[0])).as_scalar();
                let roughness = inp(
                    2,
                    SocketVal::Scalar(if node.props[1].abs() < 1e-6 {
                        0.5
                    } else {
                        node.props[1]
                    }),
                )
                .as_scalar();
                let ior = inp(
                    3,
                    SocketVal::Scalar(if node.props[2].abs() < 1e-6 {
                        1.45
                    } else {
                        node.props[2]
                    }),
                )
                .as_scalar();
                let alpha = inp(
                    4,
                    SocketVal::Scalar(if node.props[3].abs() < 1e-6 {
                        1.0
                    } else {
                        node.props[3]
                    }),
                )
                .as_scalar();
                let emission = inp(6, SocketVal::Color([0.0, 0.0, 0.0, 1.0])).as_color();
                let em_str = inp(7, SocketVal::Scalar(0.0)).as_scalar();
                let ao = inp(13, SocketVal::Scalar(1.0)).as_scalar();
                result.base_color = [base[0], base[1], base[2], alpha];
                result.metallic = metallic.clamp(0.0, 1.0);
                result.roughness = roughness.clamp(0.045, 1.0);
                result.ior = ior.max(1.0);
                result.alpha = alpha.clamp(0.0, 1.0);
                result.emissive = [
                    emission[0] * em_str,
                    emission[1] * em_str,
                    emission[2] * em_str,
                ];
                result.emission_strength = em_str;
                result.occlusion_strength = ao.clamp(0.0, 1.0);
                // Texture wiring from linked image/noise nodes
                for link in &graph.links {
                    if link.to != id {
                        continue;
                    }
                    if let Some(src) = node_map.get(&link.from) {
                        match src.kind {
                            ShaderNodeKind::ImageTexture
                            | ShaderNodeKind::Noise
                            | ShaderNodeKind::Voronoi
                            | ShaderNodeKind::Wave
                            | ShaderNodeKind::Checker
                            | ShaderNodeKind::Brick
                            | ShaderNodeKind::Magic
                            | ShaderNodeKind::Gradient
                            | ShaderNodeKind::WhiteNoise => {
                                if let Some(&ti) = proc_tex_index.get(&src.id) {
                                    match link.to_socket {
                                        0 => result.base_color_tex = Some(ti),
                                        5 => result.normal_tex = Some(ti),
                                        13 => result.occlusion_tex = Some(ti),
                                        6 => result.emissive_tex = Some(ti),
                                        _ => {}
                                    }
                                }
                            }
                            ShaderNodeKind::NormalMap => {
                                if let Some(ti) = src.image_index {
                                    result.normal_tex = Some(ti);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                cache.insert((id, 0), SocketVal::Color(result.base_color));
            }
            ShaderNodeKind::Output | ShaderNodeKind::LightOutput | ShaderNodeKind::WorldOutput => {
                if let Some(s) = find_input(graph, id, 0, &cache) {
                    cache.insert((id, 0), s);
                }
                if let Some(SocketVal::Scalar(d)) = find_input(graph, id, 1, &cache) {
                    result.volume_density = d;
                }
                if let Some(SocketVal::Scalar(d)) = find_input(graph, id, 2, &cache) {
                    result.displacement_scale = d;
                } else if let Some(SocketVal::Vector(v)) = find_input(graph, id, 2, &cache) {
                    result.displacement_scale = v[0];
                }
            }
            ShaderNodeKind::AovOutput => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                cache.insert((id, 0), SocketVal::Color(c));
            }
            ShaderNodeKind::VolumeAbsorption
            | ShaderNodeKind::VolumeScatter
            | ShaderNodeKind::VolumeCoefficients
            | ShaderNodeKind::PrincipledVolume => {
                let dens = inp(1, SocketVal::Scalar(node.value)).as_scalar();
                result.volume_density = dens;
                cache.insert(
                    (id, 0),
                    SocketVal::Color(inp(0, SocketVal::Color(node.color)).as_color()),
                );
            }
            ShaderNodeKind::Switch | ShaderNodeKind::IndexSwitch | ShaderNodeKind::MenuSwitch => {
                let sw = inp(0, SocketVal::Scalar(0.0)).as_scalar();
                let a = inp(1, SocketVal::Color([0.0, 0.0, 0.0, 1.0])).as_color();
                let b = inp(2, SocketVal::Color([1.0, 1.0, 1.0, 1.0])).as_color();
                cache.insert((id, 0), SocketVal::Color(if sw >= 0.5 { b } else { a }));
            }
            ShaderNodeKind::Geometry => {
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 0.0]));
                cache.insert((id, 1), SocketVal::Vector([0.0, 0.0, 1.0]));
                cache.insert((id, 2), SocketVal::Vector([1.0, 0.0, 0.0]));
            }
            ShaderNodeKind::CameraData => {
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 1.0]));
                cache.insert((id, 1), SocketVal::Scalar(1.0));
                cache.insert((id, 2), SocketVal::Scalar(1.0));
            }
            ShaderNodeKind::LightPath => {
                cache.insert((id, 0), SocketVal::Scalar(1.0));
                cache.insert((id, 1), SocketVal::Scalar(0.0));
                cache.insert((id, 2), SocketVal::Scalar(0.0));
                cache.insert((id, 3), SocketVal::Scalar(0.0));
                cache.insert((id, 4), SocketVal::Scalar(0.0));
            }
            ShaderNodeKind::ObjectInfo => {
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 0.0]));
                cache.insert((id, 1), SocketVal::Color(node.color));
                cache.insert((id, 2), SocketVal::Scalar(0.0));
                cache.insert((id, 3), SocketVal::Scalar(0.0));
                cache.insert((id, 4), SocketVal::Scalar(node.value));
            }
            ShaderNodeKind::ShaderToRgb => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                cache.insert((id, 0), SocketVal::Color(c));
                cache.insert((id, 1), SocketVal::Scalar(c[3]));
            }
            ShaderNodeKind::LightFalloff => {
                let s = inp(0, SocketVal::Scalar(1.0)).as_scalar();
                cache.insert((id, 0), SocketVal::Scalar(s));
                cache.insert((id, 1), SocketVal::Scalar(s));
                cache.insert((id, 2), SocketVal::Scalar(s));
            }
            ShaderNodeKind::Displacement | ShaderNodeKind::VectorDisplacement => {
                let h = inp(0, SocketVal::Scalar(node.value)).as_scalar();
                result.displacement_scale = h;
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, h]));
            }
            ShaderNodeKind::AmbientOcclusion => {
                cache.insert((id, 0), SocketVal::Color([1.0, 1.0, 1.0, 1.0]));
            }
            ShaderNodeKind::Wireframe => {
                cache.insert((id, 0), SocketVal::Scalar(0.0));
            }
            ShaderNodeKind::Attribute | ShaderNodeKind::ColorAttribute => {
                cache.insert((id, 0), SocketVal::Color(node.color));
            }
            ShaderNodeKind::SceneTime => {
                cache.insert((id, 0), SocketVal::Scalar(time_sec));
                cache.insert((id, 1), SocketVal::Scalar(frame));
            }
            ShaderNodeKind::Group => {
                cache.insert((id, 0), SocketVal::Color(node.color));
            }
            _ => {
                // Generic: first output from color or value
                if !outs.is_empty() {
                    match outs[0].ty {
                        SocketType::Color | SocketType::Shader => {
                            cache.insert((id, 0), SocketVal::Color(node.color));
                        }
                        SocketType::Vector => {
                            cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 1.0]));
                        }
                        _ => {
                            cache.insert((id, 0), SocketVal::Scalar(node.value));
                        }
                    }
                }
            }
        }
        let _ = ins; // silence
    }

    // Prefer Output surface
    for n in &graph.nodes {
        if matches!(
            n.kind,
            ShaderNodeKind::Output | ShaderNodeKind::WorldOutput | ShaderNodeKind::LightOutput
        ) {
            if let Some(v) = cache.get(&(n.id, 0)) {
                let c = v.as_color();
                result.base_color = c;
                break;
            }
        }
    }
    for n in &graph.nodes {
        if n.kind == ShaderNodeKind::Principled {
            // already applied during eval
            break;
        }
    }
    result
}

#[derive(Clone, Copy)]
struct GraphPoint {
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    occlusion: f32,
    normal: [f32; 3],
    bump_height: f32,
    bump_scale: f32,
}

impl GraphPoint {
    fn from_material(material: &Material) -> Self {
        Self {
            base_color: material.base_color,
            metallic: material.metallic,
            roughness: material.roughness,
            emissive: material.emissive,
            occlusion: material.occlusion_strength,
            normal: [0.0, 0.0, 1.0],
            bump_height: 0.0,
            bump_scale: 1.0,
        }
    }
}

fn point_input(
    links: &HashMap<(u64, u8), (u64, u8)>,
    node: &ShaderNode,
    socket: u8,
    cache: &HashMap<(u64, u8), SocketVal>,
    fallback: SocketVal,
) -> SocketVal {
    links
        .get(&(node.id, socket))
        .and_then(|&(from, from_socket)| cache.get(&(from, from_socket)).copied())
        .or_else(|| {
            if node.uses_implicit_coordinates(socket as usize) {
                None
            } else if node.input_defaults.is_empty() {
                point_socket_default_val(node, socket)
            } else {
                socket_default_val(node, socket)
            }
        })
        .unwrap_or(fallback)
}

fn point_socket_default_val(node: &ShaderNode, socket: u8) -> Option<SocketVal> {
    let def = node.kind.inputs().get(socket as usize)?;
    let value = node.socket_default(socket as usize);
    Some(match def.ty {
        SocketType::Float => SocketVal::Scalar(value[0]),
        SocketType::Color | SocketType::Shader => SocketVal::Color(value),
        SocketType::Vector => SocketVal::Vector([value[0], value[1], value[2]]),
        SocketType::String => return None,
    })
}

fn input_link_map(graph: &ShaderGraph) -> HashMap<(u64, u8), (u64, u8)> {
    graph
        .links
        .iter()
        .map(|link| ((link.to, link.to_socket), (link.from, link.from_socket)))
        .collect()
}

fn evaluate_graph_point(
    graph: &ShaderGraph,
    links: &HashMap<(u64, u8), (u64, u8)>,
    order: &[u64],
    node_map: &HashMap<u64, &ShaderNode>,
    textures: &[TextureData],
    material: &Material,
    u: f32,
    v: f32,
    cache: &mut HashMap<(u64, u8), SocketVal>,
) -> GraphPoint {
    let mut point = GraphPoint::from_material(material);
    cache.clear();
    for &id in order {
        let Some(node) = node_map.get(&id) else {
            continue;
        };
        let inp = |socket, fallback| point_input(links, node, socket, &cache, fallback);
        match node.kind {
            ShaderNodeKind::Value => {
                cache.insert((id, 0), SocketVal::Scalar(node.value));
            }
            ShaderNodeKind::Rgb => {
                cache.insert((id, 0), SocketVal::Color(node.color));
            }
            ShaderNodeKind::TexCoord | ShaderNodeKind::UvMap => {
                cache.insert((id, 0), SocketVal::Vector([u, v, 0.0]));
            }
            ShaderNodeKind::Mapping => {
                let p = inp(0, SocketVal::Vector([u, v, 0.0])).as_vector();
                let loc = inp(1, SocketVal::Vector([0.0; 3])).as_vector();
                let rot = inp(2, SocketVal::Vector([0.0; 3])).as_vector();
                let legacy_scale = [
                    if node.value.abs() < 1e-6 {
                        1.0
                    } else {
                        node.value
                    },
                    if node.props[0].abs() < 1e-6 {
                        1.0
                    } else {
                        node.props[0]
                    },
                    if node.props[1].abs() < 1e-6 {
                        1.0
                    } else {
                        node.props[1]
                    },
                ];
                let scale = inp(3, SocketVal::Vector(legacy_scale)).as_vector();
                let p = rotate_euler_xyz([p[0] * scale[0], p[1] * scale[1], p[2] * scale[2]], rot);
                cache.insert(
                    (id, 0),
                    SocketVal::Vector([p[0] + loc[0], p[1] + loc[1], p[2] + loc[2]]),
                );
            }
            ShaderNodeKind::ImageTexture | ShaderNodeKind::Environment => {
                let uv = inp(0, SocketVal::Vector([u, v, 0.0])).as_vector();
                let color = node
                    .image_index
                    .and_then(|index| textures.get(index))
                    .map(|texture| sample_tex(texture, uv[0], uv[1]))
                    .unwrap_or([1.0; 4]);
                cache.insert((id, 0), SocketVal::Color(color));
                cache.insert((id, 1), SocketVal::Scalar(color[3]));
            }
            ShaderNodeKind::Noise => {
                let p = inp(0, SocketVal::Vector([u, v, 0.0])).as_vector();
                let w = inp(1, SocketVal::Scalar(0.0)).as_scalar();
                let scale = inp(2, SocketVal::Scalar(tex_scale(node))).as_scalar();
                let detail = inp(3, SocketVal::Scalar(2.0)).as_scalar();
                let roughness = inp(4, SocketVal::Scalar(0.5)).as_scalar();
                let lacunarity = inp(5, SocketVal::Scalar(2.0)).as_scalar();
                let distortion = inp(6, SocketVal::Scalar(0.0)).as_scalar();
                let (color, fac) = noise_texture(
                    p, w, node.mode, scale, detail, roughness, lacunarity, distortion,
                );
                cache.insert((id, 0), SocketVal::Scalar(fac));
                cache.insert((id, 1), SocketVal::Color(color));
            }
            ShaderNodeKind::Voronoi => {
                let p = inp(0, SocketVal::Vector([u, v, 0.0])).as_vector();
                let scale = inp(2, SocketVal::Scalar(tex_scale(node))).as_scalar();
                let randomness = inp(7, SocketVal::Scalar(1.0)).as_scalar();
                let smoothness = inp(6, SocketVal::Scalar(0.5)).as_scalar();
                let (distance, color, position, radius) = voronoi_sample(
                    p[0] * scale,
                    p[1] * scale,
                    randomness,
                    smoothness,
                    node.mode,
                );
                cache.insert((id, 0), SocketVal::Scalar(distance));
                cache.insert((id, 1), SocketVal::Color(color));
                cache.insert((id, 2), SocketVal::Vector(position));
                cache.insert((id, 3), SocketVal::Scalar(0.0));
                cache.insert((id, 4), SocketVal::Scalar(radius));
            }
            ShaderNodeKind::Brick => {
                let p = inp(0, SocketVal::Vector([u, v, 0.0])).as_vector();
                let c1 = inp(1, SocketVal::Color([0.8, 0.8, 0.8, 1.0])).as_color();
                let c2 = inp(2, SocketVal::Color([0.2, 0.2, 0.2, 1.0])).as_color();
                let mortar = inp(3, SocketVal::Color([0.0, 0.0, 0.0, 1.0])).as_color();
                let scale = inp(4, SocketVal::Scalar(tex_scale(node)))
                    .as_scalar()
                    .max(0.01);
                let mortar_size = inp(5, SocketVal::Scalar(0.02)).as_scalar().clamp(0.0, 0.5);
                let mortar_smooth = inp(6, SocketVal::Scalar(0.1)).as_scalar().clamp(0.0, 1.0);
                let bias = inp(7, SocketVal::Scalar(0.0)).as_scalar();
                let width = inp(8, SocketVal::Scalar(0.5)).as_scalar().max(0.05);
                let row_height = inp(9, SocketVal::Scalar(0.25)).as_scalar().max(0.05);
                let (offset, offset_frequency, squash, squash_frequency) = brick_controls(node);
                let (color, mortar_factor) = brick_texture(
                    p,
                    scale,
                    c1,
                    c2,
                    mortar,
                    mortar_size,
                    mortar_smooth,
                    bias,
                    width,
                    row_height,
                    offset,
                    offset_frequency,
                    squash,
                    squash_frequency,
                );
                cache.insert((id, 0), SocketVal::Color(color));
                cache.insert((id, 1), SocketVal::Scalar(mortar_factor));
            }
            ShaderNodeKind::Checker => {
                let p = inp(0, SocketVal::Vector([u, v, 0.0])).as_vector();
                let c1 = inp(1, SocketVal::Color([1.0; 4])).as_color();
                let c2 = inp(2, SocketVal::Color([0.0, 0.0, 0.0, 1.0])).as_color();
                let scale = inp(3, SocketVal::Scalar(tex_scale(node))).as_scalar();
                let fac = if ((p[0] * scale).floor() as i32 + (p[1] * scale).floor() as i32)
                    .rem_euclid(2)
                    == 0
                {
                    1.0
                } else {
                    0.0
                };
                cache.insert((id, 0), SocketVal::Color(if fac > 0.5 { c1 } else { c2 }));
                cache.insert((id, 1), SocketVal::Scalar(fac));
            }
            ShaderNodeKind::Gradient => {
                let p = inp(0, SocketVal::Vector([u, v, 0.0])).as_vector();
                let fac = gradient_value(p[0], p[1], node.mode);
                cache.insert((id, 0), SocketVal::Scalar(fac));
                cache.insert((id, 1), SocketVal::Color([fac, fac, fac, 1.0]));
            }
            ShaderNodeKind::Wave => {
                let p = inp(0, SocketVal::Vector([u, v, 0.0])).as_vector();
                let scale = inp(1, SocketVal::Scalar(tex_scale(node))).as_scalar();
                let distortion = inp(2, SocketVal::Scalar(0.0)).as_scalar();
                let detail = inp(3, SocketVal::Scalar(2.0)).as_scalar();
                let detail_scale = inp(4, SocketVal::Scalar(1.0)).as_scalar();
                let detail_roughness = inp(5, SocketVal::Scalar(0.5)).as_scalar();
                let phase = inp(6, SocketVal::Scalar(0.0)).as_scalar();
                let fac = wave_value(
                    p,
                    scale,
                    distortion,
                    detail,
                    detail_scale,
                    detail_roughness,
                    phase,
                    node.mode,
                    node.props[8].round().clamp(0.0, 3.0) as u8,
                    node.props[9].round().clamp(0.0, 3.0) as u8,
                    node.props[10].round().clamp(0.0, 2.0) as u8,
                );
                cache.insert((id, 0), SocketVal::Scalar(fac));
                cache.insert((id, 1), SocketVal::Color([fac, fac, fac, 1.0]));
            }
            ShaderNodeKind::Magic => {
                let p = inp(0, SocketVal::Vector([u, v, 0.0])).as_vector();
                let scale = inp(1, SocketVal::Scalar(tex_scale(node))).as_scalar();
                let distortion = inp(2, SocketVal::Scalar(1.0)).as_scalar();
                let (color, fac) = magic_texture(p, scale, distortion, magic_depth(node));
                cache.insert((id, 0), SocketVal::Scalar(fac));
                cache.insert((id, 1), SocketVal::Color(color));
            }
            ShaderNodeKind::Gabor => {
                let p = inp(0, SocketVal::Vector([u, v, 0.0])).as_vector();
                let scale = inp(1, SocketVal::Scalar(5.0)).as_scalar();
                let frequency = inp(2, SocketVal::Scalar(2.0)).as_scalar();
                let anisotropy = inp(3, SocketVal::Scalar(1.0)).as_scalar();
                let orientation = inp(4, SocketVal::Scalar(0.0)).as_scalar();
                let fac = gabor_value(
                    p[0] * scale,
                    p[1] * scale,
                    p[2] * scale,
                    frequency,
                    anisotropy,
                    orientation,
                    node.mode == 1,
                );
                cache.insert((id, 0), SocketVal::Scalar(fac));
                cache.insert((id, 1), SocketVal::Color([fac, fac, fac, 1.0]));
            }
            ShaderNodeKind::WhiteNoise => {
                let p = inp(0, SocketVal::Vector([u, v, 0.0])).as_vector();
                let w = inp(1, SocketVal::Scalar(0.0)).as_scalar();
                let (fac, color) = white_noise_texture(p, w, node.mode);
                cache.insert((id, 0), SocketVal::Scalar(fac));
                cache.insert((id, 1), SocketVal::Color(color));
            }
            ShaderNodeKind::ColorRamp => {
                let fac = inp(0, SocketVal::Scalar(0.5)).as_scalar();
                let color = color_ramp_from_node(node, fac);
                cache.insert((id, 0), SocketVal::Color(color));
                cache.insert((id, 1), SocketVal::Scalar(color[3]));
            }
            ShaderNodeKind::Mix => {
                let fac = inp(0, SocketVal::Scalar(0.5)).as_scalar();
                let a = inp(1, SocketVal::Color([0.0, 0.0, 0.0, 1.0])).as_color();
                let b = inp(2, SocketVal::Color([1.0; 4])).as_color();
                cache.insert(
                    (id, 0),
                    SocketVal::Color(mix_color_mode(a, b, fac, node.mode)),
                );
            }
            ShaderNodeKind::Math => {
                let a = inp(0, SocketVal::Scalar(node.value)).as_scalar();
                let b = inp(1, SocketVal::Scalar(node.props[0])).as_scalar();
                cache.insert((id, 0), SocketVal::Scalar(math_op(node.mode, a, b)));
            }
            ShaderNodeKind::Gamma => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                let gamma = inp(1, SocketVal::Scalar(1.0)).as_scalar().max(0.01);
                cache.insert(
                    (id, 0),
                    SocketVal::Color([
                        c[0].max(0.0).powf(1.0 / gamma),
                        c[1].max(0.0).powf(1.0 / gamma),
                        c[2].max(0.0).powf(1.0 / gamma),
                        c[3],
                    ]),
                );
            }
            ShaderNodeKind::InvertColor => {
                let fac = inp(0, SocketVal::Scalar(1.0)).as_scalar();
                let c = inp(1, SocketVal::Color(node.color)).as_color();
                cache.insert(
                    (id, 0),
                    SocketVal::Color(mix4(c, [1.0 - c[0], 1.0 - c[1], 1.0 - c[2], c[3]], fac)),
                );
            }
            ShaderNodeKind::BrightContrast => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                let bright = inp(1, SocketVal::Scalar(0.0)).as_scalar();
                let contrast = inp(2, SocketVal::Scalar(0.0)).as_scalar();
                let f = |x: f32| ((x - 0.5) * (1.0 + contrast) + 0.5 + bright).clamp(0.0, 1.0);
                cache.insert((id, 0), SocketVal::Color([f(c[0]), f(c[1]), f(c[2]), c[3]]));
            }
            ShaderNodeKind::Bump => {
                let height = inp(2, SocketVal::Scalar(node.value)).as_scalar();
                let strength = inp(0, SocketVal::Scalar(1.0)).as_scalar();
                let distance = inp(1, SocketVal::Scalar(1.0)).as_scalar();
                point.bump_height = height;
                point.bump_scale = strength * distance;
                cache.insert((id, 0), SocketVal::Vector([0.0, 0.0, 1.0]));
            }
            ShaderNodeKind::NormalMap => {
                let c = links
                    .get(&(id, 1))
                    .and_then(|&(from, _)| node_map.get(&from).copied())
                    .filter(|source| source.kind == ShaderNodeKind::ImageTexture)
                    .and_then(|source| {
                        let uv =
                            point_input(links, source, 0, &cache, SocketVal::Vector([u, v, 0.0]))
                                .as_vector();
                        source
                            .image_index
                            .and_then(|index| textures.get(index))
                            .map(|texture| sample_tex_data(texture, uv[0], uv[1]))
                    })
                    .unwrap_or_else(|| inp(1, SocketVal::Color([0.5, 0.5, 1.0, 1.0])).as_color());
                let strength = inp(0, SocketVal::Scalar(1.0)).as_scalar();
                let mut n = [c[0] * 2.0 - 1.0, c[1] * 2.0 - 1.0, c[2] * 2.0 - 1.0];
                n = [
                    n[0] * strength,
                    n[1] * strength,
                    1.0 + (n[2] - 1.0) * strength,
                ];
                let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-8);
                n = [n[0] / length, n[1] / length, n[2] / length];
                point.normal = n;
                cache.insert((id, 0), SocketVal::Vector(n));
            }
            ShaderNodeKind::Principled => {
                let color = inp(0, SocketVal::Color(node.color)).as_color();
                let metallic = inp(1, SocketVal::Scalar(node.props[0])).as_scalar();
                let roughness = inp(2, SocketVal::Scalar(0.5)).as_scalar();
                let alpha = inp(4, SocketVal::Scalar(1.0)).as_scalar();
                let emission = inp(6, SocketVal::Color([0.0, 0.0, 0.0, 1.0])).as_color();
                let strength = inp(7, SocketVal::Scalar(0.0)).as_scalar();
                let ao = inp(13, SocketVal::Scalar(1.0)).as_scalar();
                point.base_color = [color[0], color[1], color[2], alpha];
                point.metallic = metallic;
                point.roughness = roughness;
                point.emissive = [
                    emission[0] * strength,
                    emission[1] * strength,
                    emission[2] * strength,
                ];
                point.occlusion = ao;
                if let Some(&(from, from_socket)) = links.get(&(id, 5)) {
                    if let Some(value) = cache.get(&(from, from_socket)) {
                        if let SocketVal::Vector(normal) = *value {
                            point.normal = normal;
                        }
                    }
                }
                cache.insert((id, 0), SocketVal::Color(point.base_color));
            }
            ShaderNodeKind::Output | ShaderNodeKind::WorldOutput | ShaderNodeKind::LightOutput => {
                if let Some(&(from, from_socket)) = links.get(&(id, 0)) {
                    if let Some(value) = cache.get(&(from, from_socket)).copied() {
                        cache.insert((id, 0), value);
                    }
                }
            }
            ShaderNodeKind::Emission
            | ShaderNodeKind::Background
            | ShaderNodeKind::Diffuse
            | ShaderNodeKind::Glossy
            | ShaderNodeKind::Transparent
            | ShaderNodeKind::Translucent
            | ShaderNodeKind::Sheen
            | ShaderNodeKind::Glass
            | ShaderNodeKind::Refraction
            | ShaderNodeKind::Toon
            | ShaderNodeKind::Subsurface
            | ShaderNodeKind::Metallic
            | ShaderNodeKind::Specular
            | ShaderNodeKind::Hair
            | ShaderNodeKind::Holdout => {
                let color = inp(0, SocketVal::Color(node.color)).as_color();
                cache.insert((id, 0), SocketVal::Color(color));
            }
            ShaderNodeKind::CombineXyz => {
                cache.insert(
                    (id, 0),
                    SocketVal::Vector([
                        inp(0, SocketVal::Scalar(0.0)).as_scalar(),
                        inp(1, SocketVal::Scalar(0.0)).as_scalar(),
                        inp(2, SocketVal::Scalar(0.0)).as_scalar(),
                    ]),
                );
            }
            ShaderNodeKind::SeparateXyz => {
                let p = inp(0, SocketVal::Vector([0.0; 3])).as_vector();
                for i in 0..3 {
                    cache.insert((id, i as u8), SocketVal::Scalar(p[i]));
                }
            }
            ShaderNodeKind::CombineColor => {
                let color = [
                    inp(0, SocketVal::Scalar(0.0)).as_scalar(),
                    inp(1, SocketVal::Scalar(0.0)).as_scalar(),
                    inp(2, SocketVal::Scalar(0.0)).as_scalar(),
                    1.0,
                ];
                cache.insert((id, 0), SocketVal::Color(color));
                cache.insert((id, 1), SocketVal::Scalar(1.0));
            }
            ShaderNodeKind::SeparateColor => {
                let color = inp(0, SocketVal::Color(node.color)).as_color();
                for i in 0..3 {
                    cache.insert((id, i as u8), SocketVal::Scalar(color[i]));
                }
            }
            ShaderNodeKind::RgbToBw => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                cache.insert(
                    (id, 0),
                    SocketVal::Scalar(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]),
                );
            }
            ShaderNodeKind::MapRange => {
                let value = inp(0, SocketVal::Scalar(node.value)).as_scalar();
                let from_min = inp(1, SocketVal::Scalar(0.0)).as_scalar();
                let from_max = inp(2, SocketVal::Scalar(1.0)).as_scalar();
                let to_min = inp(3, SocketVal::Scalar(0.0)).as_scalar();
                let to_max = inp(4, SocketVal::Scalar(1.0)).as_scalar();
                let t = if (from_max - from_min).abs() < 1e-8 {
                    0.0
                } else {
                    (value - from_min) / (from_max - from_min)
                };
                let t = if node.mode == 0 { t.clamp(0.0, 1.0) } else { t };
                cache.insert((id, 0), SocketVal::Scalar(to_min + t * (to_max - to_min)));
            }
            ShaderNodeKind::Clamp => {
                let value = inp(0, SocketVal::Scalar(node.value)).as_scalar();
                let min = inp(1, SocketVal::Scalar(0.0)).as_scalar();
                let max = inp(2, SocketVal::Scalar(1.0)).as_scalar();
                cache.insert((id, 0), SocketVal::Scalar(value.clamp(min, max)));
            }
            ShaderNodeKind::AmbientOcclusion => {
                let noise = noise2(u * 4.0, v * 4.0);
                let ao = (0.55 + 0.45 * noise).clamp(0.0, 1.0);
                cache.insert((id, 0), SocketVal::Color([ao, ao, ao, 1.0]));
                cache.insert((id, 1), SocketVal::Scalar(ao));
            }
            _ => {
                if let Some(output) = node.kind.outputs().first() {
                    let value = match output.ty {
                        SocketType::Color | SocketType::Shader => SocketVal::Color(node.color),
                        SocketType::Vector => SocketVal::Vector([0.0, 0.0, 1.0]),
                        _ => SocketVal::Scalar(node.value),
                    };
                    cache.insert((id, 0), value);
                }
            }
        }
    }
    for node in &graph.nodes {
        if matches!(
            node.kind,
            ShaderNodeKind::Output | ShaderNodeKind::WorldOutput | ShaderNodeKind::LightOutput
        ) {
            if let Some(value) = cache.get(&(node.id, 0)) {
                point.base_color = value.as_color();
                break;
            }
        }
    }
    point
}

pub fn evaluate_graph_ex(
    graph: &ShaderGraph,
    material: &Material,
    textures: &[TextureData],
    tex_size: u32,
    live: bool,
    frame: f32,
    time_sec: f32,
) -> GraphEvalResult {
    evaluate_graph_ex_inner(
        graph, material, textures, tex_size, live, frame, time_sec, None,
    )
    .expect("non-cancellable graph evaluation cannot be cancelled")
}

pub fn evaluate_graph_ex_cancellable(
    graph: &ShaderGraph,
    material: &Material,
    textures: &[TextureData],
    tex_size: u32,
    live: bool,
    frame: f32,
    time_sec: f32,
    cancel: &AtomicBool,
) -> Option<GraphEvalResult> {
    evaluate_graph_ex_inner(
        graph,
        material,
        textures,
        tex_size,
        live,
        frame,
        time_sec,
        Some(cancel),
    )
}

fn evaluate_graph_ex_inner(
    graph: &ShaderGraph,
    material: &Material,
    textures: &[TextureData],
    tex_size: u32,
    live: bool,
    frame: f32,
    time_sec: f32,
    cancel: Option<&AtomicBool>,
) -> Option<GraphEvalResult> {
    if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        return None;
    }
    let mut result =
        evaluate_graph_center(graph, material, textures, tex_size, true, frame, time_sec);
    if live {
        return Some(result);
    }

    let Some(principled) = graph
        .nodes
        .iter()
        .find(|node| node.kind == ShaderNodeKind::Principled)
    else {
        return Some(result);
    };
    let linked = |socket| {
        graph
            .links
            .iter()
            .any(|link| link.to == principled.id && link.to_socket == socket)
    };
    let bake_base = linked(0) || linked(4);
    let bake_metallic_roughness = linked(1) || linked(2);
    let bake_emission = linked(6) || linked(7);
    let bake_ao = linked(13);
    let normal_source = graph
        .links
        .iter()
        .find(|link| link.to == principled.id && link.to_socket == 5)
        .and_then(|link| graph.nodes.iter().find(|node| node.id == link.from));
    let bake_normal = normal_source.map_or(false, |node| {
        matches!(node.kind, ShaderNodeKind::Bump | ShaderNodeKind::NormalMap)
            || node
                .kind
                .outputs()
                .get(
                    graph
                        .links
                        .iter()
                        .find(|link| link.to == principled.id && link.to_socket == 5)
                        .unwrap()
                        .from_socket as usize,
                )
                .map_or(false, |socket| socket.ty == SocketType::Vector)
    });
    if !(bake_base || bake_metallic_roughness || bake_emission || bake_ao || bake_normal) {
        return Some(result);
    }

    let order = topo_order(graph);
    let links = input_link_map(graph);
    let node_map: HashMap<u64, &ShaderNode> =
        graph.nodes.iter().map(|node| (node.id, node)).collect();
    let size = tex_size.clamp(64, 512);
    let count = (size * size) as usize;
    let mut points = Vec::with_capacity(count);
    let mut cache = HashMap::with_capacity(graph.nodes.len() * 3);
    for y in 0..size {
        if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return None;
        }
        for x in 0..size {
            points.push(evaluate_graph_point(
                graph,
                &links,
                &order,
                &node_map,
                textures,
                material,
                (x as f32 + 0.5) / size as f32,
                (y as f32 + 0.5) / size as f32,
                &mut cache,
            ));
        }
    }
    if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        return None;
    }
    let index_for_new_texture =
        |result: &GraphEvalResult| textures.len() + result.new_textures.len();
    if bake_base {
        let mut rgba = Vec::with_capacity(count * 4);
        for point in &points {
            rgba.extend_from_slice(&[
                linear_to_srgb8(point.base_color[0]),
                linear_to_srgb8(point.base_color[1]),
                linear_to_srgb8(point.base_color[2]),
                (point.base_color[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            ]);
        }
        result.base_color_tex = Some(index_for_new_texture(&result));
        result.base_color = [1.0; 4];
        result.new_textures.push(TextureData {
            name: "__sg_BaseColor".into(),
            width: size,
            height: size,
            rgba8: rgba,
        });
    }
    if bake_metallic_roughness {
        let mut rgba = Vec::with_capacity(count * 4);
        for point in &points {
            rgba.extend_from_slice(&[
                255,
                (point.roughness.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                (point.metallic.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                255,
            ]);
        }
        result.metallic_roughness_tex = Some(index_for_new_texture(&result));
        result.metallic = 1.0;
        result.roughness = 1.0;
        result.new_textures.push(TextureData {
            name: "__sg_MetalRough".into(),
            width: size,
            height: size,
            rgba8: rgba,
        });
    }
    if bake_emission {
        let mut rgba = Vec::with_capacity(count * 4);
        for point in &points {
            rgba.extend_from_slice(&[
                linear_to_srgb8(point.emissive[0]),
                linear_to_srgb8(point.emissive[1]),
                linear_to_srgb8(point.emissive[2]),
                255,
            ]);
        }
        result.emissive_tex = Some(index_for_new_texture(&result));
        result.emissive = [1.0; 3];
        result.new_textures.push(TextureData {
            name: "__sg_Emission".into(),
            width: size,
            height: size,
            rgba8: rgba,
        });
    }
    if bake_ao {
        let mut rgba = Vec::with_capacity(count * 4);
        for point in &points {
            let value = (point.occlusion.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            rgba.extend_from_slice(&[value, value, value, 255]);
        }
        result.occlusion_tex = Some(index_for_new_texture(&result));
        result.occlusion_strength = 1.0;
        result.new_textures.push(TextureData {
            name: "__sg_AO".into(),
            width: size,
            height: size,
            rgba8: rgba,
        });
    }
    if bake_normal {
        let bump = normal_source.map_or(false, |node| node.kind == ShaderNodeKind::Bump);
        let mut rgba = Vec::with_capacity(count * 4);
        for y in 0..size {
            for x in 0..size {
                let i = (y * size + x) as usize;
                let normal = if bump {
                    let left = points[(y * size + x.saturating_sub(1)) as usize].bump_height;
                    let right = points[(y * size + (x + 1).min(size - 1)) as usize].bump_height;
                    let up = points[(y.saturating_sub(1) * size + x) as usize].bump_height;
                    let down = points[((y + 1).min(size - 1) * size + x) as usize].bump_height;
                    let gradient = [
                        (right - left) * size as f32 * 0.5 * points[i].bump_scale,
                        (down - up) * size as f32 * 0.5 * points[i].bump_scale,
                        1.0,
                    ];
                    let length =
                        (gradient[0] * gradient[0] + gradient[1] * gradient[1] + 1.0).sqrt();
                    [
                        gradient[0] / length,
                        gradient[1] / length,
                        gradient[2] / length,
                    ]
                } else {
                    points[i].normal
                };
                rgba.extend_from_slice(&[
                    ((normal[0] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                    ((normal[1] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                    ((normal[2] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                    255,
                ]);
            }
        }
        result.normal_tex = Some(index_for_new_texture(&result));
        result.normal_scale = 1.0;
        result.new_textures.push(TextureData {
            name: "__sg_Normal".into(),
            width: size,
            height: size,
            rgba8: rgba,
        });
    }
    Some(result)
}

/// Fast single-UV sample for live viewport (no texture allocation).

fn tex_scale(node: &ShaderNode) -> f32 {
    let scale_socket = match node.kind {
        ShaderNodeKind::Noise | ShaderNodeKind::Voronoi => Some(2),
        ShaderNodeKind::Wave | ShaderNodeKind::Magic | ShaderNodeKind::Gabor => Some(1),
        ShaderNodeKind::Brick => Some(4),
        ShaderNodeKind::Checker => Some(3),
        _ => None,
    };
    if let Some(scale_socket) = scale_socket {
        if let Some(value) = node.input_defaults.get(scale_socket) {
            return value[0].abs().max(0.01);
        }
    }
    if node.value.abs() > 1e-4 {
        if (node.value - 0.5).abs() < 1e-6
            && matches!(
                node.kind,
                ShaderNodeKind::Noise
                    | ShaderNodeKind::Voronoi
                    | ShaderNodeKind::Wave
                    | ShaderNodeKind::Magic
                    | ShaderNodeKind::Gabor
            )
        {
            5.0
        } else {
            node.value.abs()
        }
    } else if node.props[0].abs() > 1e-4 {
        // older UI stored Scale in props[0]
        node.props[0].abs()
    } else {
        5.0
    }
}

fn noise_texture(
    p: [f32; 3],
    w: f32,
    dimension: u8,
    scale: f32,
    detail: f32,
    roughness: f32,
    lacunarity: f32,
    distortion: f32,
) -> ([f32; 4], f32) {
    let mut coords = [p[0] * scale, p[1] * scale, p[2] * scale, w * scale];
    if distortion.abs() > 1e-6 {
        let dx = noise2(coords[0] + 17.1, coords[1] + 9.2) * 2.0 - 1.0;
        let dy = noise2(coords[0] + 3.7, coords[1] + 28.4) * 2.0 - 1.0;
        coords[0] += dx * distortion;
        coords[1] += dy * distortion;
    }
    let dimensions = dimension.min(3) as usize + 1;
    let fac = fractal_perlin(coords, dimensions, detail, roughness, lacunarity);
    let color = [
        fractal_perlin(
            [
                coords[0] + 17.1,
                coords[1] + 9.2,
                coords[2] + 3.7,
                coords[3] + 28.4,
            ],
            dimensions,
            detail,
            roughness,
            lacunarity,
        ),
        fractal_perlin(
            [
                coords[0] + 41.7,
                coords[1] + 13.3,
                coords[2] + 23.8,
                coords[3] + 7.9,
            ],
            dimensions,
            detail,
            roughness,
            lacunarity,
        ),
        fractal_perlin(
            [
                coords[0] + 5.2,
                coords[1] + 37.6,
                coords[2] + 11.4,
                coords[3] + 19.1,
            ],
            dimensions,
            detail,
            roughness,
            lacunarity,
        ),
        1.0,
    ];
    (color, fac)
}

fn fractal_perlin(
    coords: [f32; 4],
    dimensions: usize,
    detail: f32,
    roughness: f32,
    lacunarity: f32,
) -> f32 {
    let detail = detail.clamp(0.0, 15.0);
    let full_octaves = detail.floor() as usize;
    let fractional_octave = detail.fract();
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut sum = 0.0;
    let mut weight = 0.0;
    for _ in 0..=full_octaves {
        let octave_weight = amplitude;
        let sample = perlin_noise(coords.map(|coordinate| coordinate * frequency), dimensions);
        sum += sample * octave_weight;
        weight += octave_weight;
        amplitude *= roughness.clamp(0.0, 1.0);
        frequency *= lacunarity.max(1.0);
    }
    if fractional_octave > 0.0 {
        let sample = perlin_noise(coords.map(|coordinate| coordinate * frequency), dimensions);
        let octave_weight = amplitude * fractional_octave;
        sum += sample * octave_weight;
        weight += octave_weight;
    }
    if weight > 1e-8 {
        (0.5 + sum / weight * 0.5).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

fn perlin_noise(coords: [f32; 4], dimensions: usize) -> f32 {
    let dimensions = dimensions.clamp(1, 4);
    let lattice = coords.map(f32::floor);
    let fade = coords.map(|coordinate| {
        let t = coordinate - coordinate.floor();
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    });
    let mut value = 0.0;
    for corner in 0..(1 << dimensions) {
        let mut point = lattice;
        let mut offset = [0.0; 4];
        let mut interpolation_weight = 1.0;
        for axis in 0..dimensions {
            let high = corner & (1 << axis) != 0;
            if high {
                point[axis] += 1.0;
            }
            offset[axis] = coords[axis] - point[axis];
            interpolation_weight *= if high { fade[axis] } else { 1.0 - fade[axis] };
        }
        let seed = match dimensions {
            1 => hash2(point[0], 71.3),
            2 => hash2(point[0], point[1]),
            3 => hash3(point[0], point[1], point[2]),
            _ => hash4(point[0], point[1], point[2], point[3]),
        };
        let mut gradient = [0.0; 4];
        let mut length_squared = 0.0;
        for axis in 0..dimensions {
            let component = hash4(
                point[0] + 13.1 * (axis + 1) as f32,
                point[1] + seed * 31.7,
                point[2] + 19.3 * (axis + 1) as f32,
                point[3] + seed * 47.9,
            ) * 2.0
                - 1.0;
            gradient[axis] = component;
            length_squared += component * component;
        }
        let inv_length = length_squared.sqrt().max(1e-8).recip();
        let dot = (0..dimensions)
            .map(|axis| gradient[axis] * offset[axis] * inv_length)
            .sum::<f32>();
        value += dot * interpolation_weight;
    }
    (value * 2.0).clamp(-1.0, 1.0)
}

fn gradient_value(u: f32, v: f32, mode: u8) -> f32 {
    let dx = u - 0.5;
    let dy = v - 0.5;
    match mode {
        1 => u.clamp(0.0, 1.0).powi(2),
        2 => {
            let x = u.clamp(0.0, 1.0);
            x * x * (3.0 - 2.0 * x)
        }
        3 => ((u + v) * 0.5).clamp(0.0, 1.0),
        4 => (1.0 - (dx * dx + dy * dy).sqrt() * 2.0).clamp(0.0, 1.0),
        5 => (1.0 - (dx * dx + dy * dy) * 4.0).clamp(0.0, 1.0),
        6 => ((dy.atan2(dx) / std::f32::consts::TAU) + 0.5).rem_euclid(1.0),
        _ => u.clamp(0.0, 1.0),
    }
}

fn brick_noise(row: i32, brick: i32) -> f32 {
    let mut n = (row as u32)
        .wrapping_shl(16)
        .wrapping_add((brick as u32) & 0xFFFF);
    n = n.wrapping_add(1013) & 0x7FFF_FFFF;
    n = (n >> 13) ^ n;
    let nn = n
        .wrapping_mul(
            n.wrapping_mul(n)
                .wrapping_mul(60493)
                .wrapping_add(19990303),
        )
        .wrapping_add(1376312589)
        & 0x7FFF_FFFF;
    0.5 * (nn as f32 / 1_073_741_824.0)
}

fn brick_controls(node: &ShaderNode) -> (f32, i32, f32, i32) {
    let properties = if node.props[..4].iter().all(|value| *value == 0.0) {
        [0.5, 2.0, 1.0, 2.0]
    } else {
        node.props[..4].try_into().unwrap_or([0.5, 2.0, 1.0, 2.0])
    };
    (
        properties[0],
        properties[1].round() as i32,
        properties[2],
        properties[3].round() as i32,
    )
}

fn magic_depth(node: &ShaderNode) -> u8 {
    if node.props[9] > 0.5 {
        node.props[8].round().clamp(0.0, 10.0) as u8
    } else {
        node.mode.saturating_add(2).min(10)
    }
}

fn brick_texture(
    p: [f32; 3],
    scale: f32,
    color1: [f32; 4],
    color2: [f32; 4],
    mortar_color: [f32; 4],
    mortar_size: f32,
    mortar_smooth: f32,
    bias: f32,
    brick_width: f32,
    row_height: f32,
    offset_amount: f32,
    offset_frequency: i32,
    squash_amount: f32,
    squash_frequency: i32,
) -> ([f32; 4], f32) {
    let row_height = row_height.max(1e-5);
    let row = (p[1] * scale / row_height).floor() as i32;
    let mut width = brick_width.max(1e-5);
    let offset_frequency = offset_frequency.max(1);
    let squash_frequency = squash_frequency.max(1);
    if row.rem_euclid(squash_frequency) == 0 {
        width *= squash_amount.max(1e-4);
    }
    let offset = if row.rem_euclid(offset_frequency) == 0 {
        width * offset_amount
    } else {
        0.0
    };
    let x = p[0] * scale + offset;
    let brick = ((x / width).floor()) as i32;
    let local_x = x - width * brick as f32;
    let local_y = p[1] * scale - row_height * row as f32;
    let min_distance = local_x
        .min(local_y)
        .min(width - local_x)
        .min(row_height - local_y);
    let mortar_size = mortar_size.max(0.0);
    let mortar = if min_distance >= mortar_size {
        0.0
    } else if mortar_smooth <= 0.0 || mortar_size <= 0.0 {
        1.0
    } else {
        let edge = (1.0 - min_distance / mortar_size).max(0.0);
        let t = (edge / mortar_smooth).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let tint = (brick_noise(row, brick) + bias).clamp(0.0, 1.0);
    let brick_color = mix4(color1, color2, tint);
    (mix4(brick_color, mortar_color, mortar), mortar)
}

fn wave_value(
    p: [f32; 3],
    scale: f32,
    distortion: f32,
    detail: f32,
    detail_scale: f32,
    detail_roughness: f32,
    phase: f32,
    wave_type: u8,
    bands_direction: u8,
    rings_direction: u8,
    profile: u8,
) -> f32 {
    let p = p.map(|component| (component * scale + 0.000001) * 0.999999);
    let mut value = if wave_type == 0 {
        match bands_direction {
            1 => p[1] * 20.0,
            2 => p[2] * 20.0,
            3 => (p[0] + p[1] + p[2]) * 10.0,
            _ => p[0] * 20.0,
        }
    } else {
        let masked = match rings_direction {
            0 => [0.0, p[1], p[2]],
            1 => [p[0], 0.0, p[2]],
            2 => [p[0], p[1], 0.0],
            _ => p,
        };
        (masked[0] * masked[0] + masked[1] * masked[1] + masked[2] * masked[2]).sqrt() * 20.0
    } + phase;
    if distortion != 0.0 {
        let noise = noise_texture(
            p,
            0.0,
            2,
            detail_scale,
            detail,
            detail_roughness,
            2.0,
            0.0,
        )
        .1;
        value += distortion * (noise * 2.0 - 1.0);
    }
    match profile {
        1 => (value / std::f32::consts::TAU).rem_euclid(1.0),
        2 => {
            let normalized = value / std::f32::consts::TAU;
            (normalized - (normalized + 0.5).floor()).abs() * 2.0
        }
        _ => 0.5 + 0.5 * (value - std::f32::consts::FRAC_PI_2).sin(),
    }
}

fn magic_texture(p: [f32; 3], scale: f32, distortion: f32, depth: u8) -> ([f32; 4], f32) {
    let co = p.map(|component| component * scale);
    let mut x = ((co[0] + co[1] + co[2]) * 5.0).sin();
    let mut y = ((-co[0] + co[1] - co[2]) * 5.0).cos();
    let mut z = ((-co[0] - co[1] + co[2]) * 5.0).cos() * -1.0;
    if depth > 0 {
        x *= distortion;
        y *= distortion;
        z *= distortion;
        y = -(x - y + z).cos() * distortion;
        if depth > 1 {
            x = (x - y - z).cos() * distortion;
            if depth > 2 {
                z = (-x - y - z).sin() * distortion;
                if depth > 3 {
                    x = -(-x + y - z).cos() * distortion;
                    if depth > 4 {
                        y = -(-x + y + z).sin() * distortion;
                        if depth > 5 {
                            y = -(-x + y + z).cos() * distortion;
                            if depth > 6 {
                                x = (x + y + z).cos() * distortion;
                                if depth > 7 {
                                    z = (x + y - z).sin() * distortion;
                                    if depth > 8 {
                                        x = -(-x - y + z).cos() * distortion;
                                        if depth > 9 {
                                            y = (x - y + z).sin() * -distortion;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if distortion != 0.0 {
        let divisor = distortion * 2.0;
        x /= divisor;
        y /= divisor;
        z /= divisor;
    }
    let color = [0.5 - x, 0.5 - y, 0.5 - z, 1.0];
    let fac = (color[0] + color[1] + color[2]) / 3.0;
    (color, fac)
}

fn gabor_value(
    x: f32,
    y: f32,
    z: f32,
    frequency: f32,
    anisotropy: f32,
    orientation: f32,
    three_dimensional: bool,
) -> f32 {
    let (sin_o, cos_o) = orientation.sin_cos();
    let mut sum = 0.0;
    let mut weight = 0.0;
    let cx = x.floor() as i32;
    let cy = y.floor() as i32;
    let cz = z.floor() as i32;
    let z_range = if three_dimensional { -2..=2 } else { 0..=0 };
    for iz in z_range {
        for iy in -2..=2 {
            for ix in -2..=2 {
                let gx = cx + ix;
                let gy = cy + iy;
                let gz = cz + iz;
                let ox = gx as f32 + hash2(gx as f32, gy as f32 + gz as f32 * 17.0);
                let oy = gy as f32 + hash2(gx as f32 + 23.7, gy as f32 + gz as f32 * 13.0 + 9.2);
                let oz = gz as f32 + hash2(gx as f32 + 37.1, gy as f32 + gz as f32 * 7.0 + 4.4);
                let dx = x - ox;
                let dy = y - oy;
                let dz = z - oz;
                let along = dx * cos_o + dy * sin_o;
                let across = -dx * sin_o + dy * cos_o;
                let envelope = (-(along * along
                    + across * across * anisotropy.max(0.01)
                    + if three_dimensional {
                        dz * dz * anisotropy.max(0.01)
                    } else {
                        0.0
                    }))
                .exp();
                let phase = hash2(gx as f32 + 11.1, gy as f32 + gz as f32 * 19.0 + 37.7)
                    * std::f32::consts::TAU;
                let wave = (along * frequency * std::f32::consts::TAU + phase).cos();
                sum += envelope * wave;
                weight += envelope;
            }
        }
    }
    (0.5 + 0.5 * sum / weight.max(1e-8)).clamp(0.0, 1.0)
}

fn voronoi_sample(
    x: f32,
    y: f32,
    randomness: f32,
    smoothness: f32,
    mode: u8,
) -> (f32, [f32; 4], [f32; 3], f32) {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let randomness = randomness.clamp(0.0, 1.0);
    let mut nearest = (f32::MAX, [0.0; 3], [0.0; 3]);
    let mut second = (f32::MAX, [0.0; 3], [0.0; 3]);
    let smoothness = smoothness.clamp(0.0, 1.0);
    let mut smooth = f32::MAX;
    let mut smooth_color = [0.0; 3];
    let mut smooth_position = [0.0; 3];
    for dy in -2..=2 {
        for dx in -2..=2 {
            let cx = ix + dx;
            let cy = iy + dy;
            let px = cx as f32 + 0.5 + (hash2(cx as f32 + 1.7, cy as f32 + 3.1) - 0.5) * randomness;
            let py = cy as f32 + 0.5 + (hash2(cx as f32 + 5.3, cy as f32 + 2.9) - 0.5) * randomness;
            let distance = (x - px).hypot(y - py);
            let color = [
                hash2(cx as f32, cy as f32),
                hash2(cx as f32 + 9.1, cy as f32),
                hash2(cx as f32, cy as f32 + 4.4),
            ];
            if smoothness <= 1e-8 {
                if distance < smooth {
                    smooth = distance;
                    smooth_color = color;
                    smooth_position = [px, py, 0.0];
                }
            } else {
                let h = (0.5 + 0.5 * (distance - smooth) / smoothness).clamp(0.0, 1.0);
                smooth = distance * (1.0 - h) + smooth * h
                    - smoothness * h * (1.0 - h);
                for component in 0..3 {
                    smooth_color[component] =
                        color[component] * (1.0 - h) + smooth_color[component] * h;
                    let feature_position = if component == 0 { px } else if component == 1 { py } else { 0.0 };
                    smooth_position[component] =
                        feature_position * (1.0 - h) + smooth_position[component] * h;
                }
            }
            if distance < nearest.0 {
                second = nearest;
                nearest = (distance, color, [px, py, 0.0]);
            } else if distance < second.0 {
                second = (distance, color, [px, py, 0.0]);
            }
        }
    }
    let radius = (nearest.2[0] - second.2[0]).hypot(nearest.2[1] - second.2[1]) * 0.5;
    let edge_distance = ((second.0 * second.0 - nearest.0 * nearest.0)
        / (2.0 * (nearest.2[0] - second.2[0]).hypot(nearest.2[1] - second.2[1]).max(1e-8)))
    .max(0.0);
    let (value, color, position) = match mode {
        1 => (second.0, second.1, second.2),
        2 => (smooth.max(0.0), smooth_color, smooth_position),
        3 => (edge_distance, nearest.1, nearest.2),
        _ => (nearest.0, nearest.1, nearest.2),
    };
    let value = match mode {
        4 => nearest.0,
        _ => value,
    }
    .clamp(0.0, 1.0);
    (
        value,
        [color[0], color[1], color[2], 1.0],
        position,
        radius,
    )
}

fn proc_sample(
    kind: ShaderNodeKind,
    mode: u8,
    value: f32,
    props: &[f32; 16],
    color: [f32; 4],
    u: f32,
    v: f32,
) -> ([f32; 4], f32) {
    let scale = if value.abs() < 1e-4 { 5.0 } else { value.abs() };
    match kind {
        ShaderNodeKind::Noise => {
            let detail = if props[0] > 0.0 { props[0] } else { 4.0 };
            noise_texture(
                [u, v, 0.0],
                props[5],
                mode,
                scale,
                detail,
                if props[1] > 0.0 { props[1] } else { 0.5 },
                if props[2] > 0.0 { props[2] } else { 2.0 },
                props[3],
            )
        }
        ShaderNodeKind::Magic => magic_texture(
            [u, v, 0.0],
            scale,
            if props[0] == 0.0 { 1.0 } else { props[0] },
            if props[9] > 0.5 {
                props[8].round().clamp(0.0, 10.0) as u8
            } else {
                mode.saturating_add(2).min(10)
            },
        ),
        ShaderNodeKind::Gabor => {
            let fac = gabor_value(
                (u - 0.5) * scale,
                (v - 0.5) * scale,
                0.0,
                if props[0] > 0.0 { props[0] } else { 2.0 },
                if props[1] > 0.0 { props[1] } else { 1.0 },
                props[2],
                mode == 1,
            );
            ([fac, fac, fac, 1.0], fac)
        }
        ShaderNodeKind::Voronoi => {
            let (distance, color, _, _) = voronoi_sample(
                u * scale,
                v * scale,
                if props[0] > 0.0 { props[0] } else { 1.0 },
                if props[1] > 0.0 { props[1] } else { 0.5 },
                mode,
            );
            (color, distance)
        }
        ShaderNodeKind::WhiteNoise => {
            let (fac, color) = white_noise_texture([u, v, 0.0], props[0], mode);
            (color, fac)
        }
        ShaderNodeKind::Checker => {
            let cx = (u * scale).floor() as i32;
            let cy = (v * scale).floor() as i32;
            let fac = if (cx + cy) % 2 == 0 { 1.0 } else { 0.0 };
            let c = if fac > 0.5 {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.0, 0.0, 0.0, 1.0]
            };
            let _ = color;
            (c, fac)
        }
        ShaderNodeKind::Brick => {
            let (color, factor) = brick_texture(
                [u, v, 0.0],
                scale,
                [0.8, 0.8, 0.8, 1.0],
                [0.2, 0.2, 0.2, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                0.02,
                0.0,
                0.0,
                0.5,
                0.25,
                if props[..4].iter().all(|value| *value == 0.0) {
                    0.5
                } else {
                    props[0]
                },
                if props[1] > 0.0 { props[1].round() as i32 } else { 2 },
                if props[2] > 0.0 { props[2] } else { 1.0 },
                if props[3] > 0.0 { props[3].round() as i32 } else { 2 },
            );
            (color, factor)
        }
        ShaderNodeKind::Gradient => {
            let t = gradient_value(u, v, mode);
            ([t, t, t, 1.0], t)
        }
        ShaderNodeKind::Wave => {
            let t = wave_value(
                [u, v, 0.0],
                scale,
                props[0],
                if props[1] > 0.0 { props[1] } else { 2.0 },
                if props[3] > 0.0 { props[3] } else { 1.0 },
                if props[4] > 0.0 { props[4] } else { 0.5 },
                props[2],
                mode,
                props[8].round().clamp(0.0, 3.0) as u8,
                props[9].round().clamp(0.0, 3.0) as u8,
                props[10].round().clamp(0.0, 2.0) as u8,
            );
            ([t, t, t, 1.0], t)
        }
        ShaderNodeKind::Sky => (
            [
                0.45 + 0.2 * (1.0 - v),
                0.55 + 0.15 * (1.0 - v),
                0.85 - 0.1 * v,
                1.0,
            ],
            1.0,
        ),
        ShaderNodeKind::Ies => {
            let dx = u - 0.5;
            let dy = v - 0.5;
            let t = ((1.0 - (dx * dx + dy * dy).sqrt() * 2.0).max(0.0).powf(2.0)).clamp(0.0, 1.0);
            ([t, t, t, 1.0], t)
        }
        _ => ([0.5, 0.5, 0.5, 1.0], 0.5),
    }
}

fn voronoi_ex(
    x: f32,
    y: f32,
    randomness: f32,
    smoothness: f32,
    mode: u8,
) -> (f32, [f32; 3]) {
    let (distance, color, _, _) = voronoi_sample(x, y, randomness, smoothness, mode);
    (distance, [color[0], color[1], color[2]])
}

fn math_op(mode: u8, a: f32, b: f32) -> f32 {
    match mode {
        0 => a + b,
        1 => a - b,
        2 => a * b,
        3 => {
            if b.abs() > 1e-8 {
                a / b
            } else {
                0.0
            }
        }
        4 => a.powf(b),
        5 => {
            if a > 0.0 && b > 0.0 {
                a.log(b)
            } else {
                0.0
            }
        }
        6 => a.max(0.0).sqrt(),
        7 => {
            if a > 1e-8 {
                1.0 / a.sqrt()
            } else {
                0.0
            }
        }
        8 => a.abs(),
        9 => a.exp(),
        10 => a.min(b),
        11 => a.max(b),
        12 => {
            if a < b {
                1.0
            } else {
                0.0
            }
        }
        13 => {
            if a > b {
                1.0
            } else {
                0.0
            }
        }
        14 => a.signum(),
        15 => {
            if (a - b).abs() < 1e-5 {
                1.0
            } else {
                0.0
            }
        }
        16 => {
            // smooth min
            let k = b.max(1e-5);
            let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
            a * h + b * (1.0 - h) - k * h * (1.0 - h)
        }
        17 => {
            let k = b.max(1e-5);
            let h = (0.5 + 0.5 * (a - b) / k).clamp(0.0, 1.0);
            a * h + b * (1.0 - h) + k * h * (1.0 - h)
        }
        18 => a.round(),
        19 => a.floor(),
        20 => a.ceil(),
        21 => a.trunc(),
        22 => a.fract(),
        23 => {
            if b.abs() > 1e-8 {
                a % b
            } else {
                0.0
            }
        }
        24 => {
            // wrap
            let r = b;
            if r.abs() < 1e-8 {
                a
            } else {
                a - r * (a / r).floor()
            }
        }
        25 => {
            // snap
            if b.abs() < 1e-8 {
                a
            } else {
                (a / b).floor() * b
            }
        }
        26 => {
            // ping-pong
            let r = b.abs().max(1e-8);
            let t = a.rem_euclid(r * 2.0);
            if t < r {
                t
            } else {
                r * 2.0 - t
            }
        }
        27 => a.sin(),
        28 => a.cos(),
        29 => a.tan(),
        30 => a.clamp(-1.0, 1.0).asin(),
        31 => a.clamp(-1.0, 1.0).acos(),
        32 => a.atan(),
        33 => a.atan2(b),
        34 => a.sinh(),
        35 => a.cosh(),
        36 => a.tanh(),
        37 => a * std::f32::consts::PI / 180.0,
        38 => a * 180.0 / std::f32::consts::PI,
        _ => a + b,
    }
}

/// Tiny expression evaluator: supports a, b, + - * / ( ), numbers.
fn eval_simple_expr(expr: &str, a: f32, b: f32) -> f32 {
    let s = expr.trim().to_lowercase().replace(" ", "");
    if s.is_empty() {
        return a;
    }
    // Replace identifiers
    let mut t = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_alphabetic() {
            let mut j = i;
            while j < chars.len() && chars[j].is_ascii_alphanumeric() {
                j += 1;
            }
            let name: String = chars[i..j].iter().collect();
            let val = match name.as_str() {
                "a" | "x" | "value" => a,
                "b" | "y" => b,
                "pi" => std::f32::consts::PI,
                _ => 0.0,
            };
            t.push_str(&format!("({})", val));
            i = j;
        } else {
            t.push(chars[i]);
            i += 1;
        }
    }
    eval_expr_tokens(&t).unwrap_or(a)
}

fn eval_expr_tokens(s: &str) -> Option<f32> {
    // Recursive descent: expr = term (+|-) term, term = factor (*|/) factor, factor = number | (expr)
    struct P<'a> {
        s: &'a [u8],
        i: usize,
    }
    impl<'a> P<'a> {
        fn peek(&self) -> Option<u8> {
            self.s.get(self.i).copied()
        }
        fn get(&mut self) -> Option<u8> {
            let c = self.peek()?;
            self.i += 1;
            Some(c)
        }
        fn skip(&mut self) {
            while matches!(self.peek(), Some(b' ')) {
                self.i += 1;
            }
        }
        fn number(&mut self) -> Option<f32> {
            self.skip();
            let start = self.i;
            if self.peek() == Some(b'-') && start == self.i {
                self.i += 1;
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == b'.') {
                self.i += 1;
            }
            if self.i == start {
                return None;
            }
            std::str::from_utf8(&self.s[start..self.i])
                .ok()?
                .parse()
                .ok()
        }
        fn factor(&mut self) -> Option<f32> {
            self.skip();
            if self.peek() == Some(b'(') {
                self.get();
                let v = self.expr()?;
                self.skip();
                if self.get() != Some(b')') {
                    return None;
                }
                return Some(v);
            }
            if self.peek() == Some(b'-') {
                self.get();
                return Some(-self.factor()?);
            }
            self.number()
        }
        fn term(&mut self) -> Option<f32> {
            let mut v = self.factor()?;
            loop {
                self.skip();
                match self.peek() {
                    Some(b'*') => {
                        self.get();
                        v *= self.factor()?;
                    }
                    Some(b'/') => {
                        self.get();
                        let d = self.factor()?;
                        if d.abs() > 1e-12 {
                            v /= d;
                        }
                    }
                    _ => break,
                }
            }
            Some(v)
        }
        fn expr(&mut self) -> Option<f32> {
            let mut v = self.term()?;
            loop {
                self.skip();
                match self.peek() {
                    Some(b'+') => {
                        self.get();
                        v += self.term()?;
                    }
                    Some(b'-') => {
                        self.get();
                        v -= self.term()?;
                    }
                    _ => break,
                }
            }
            Some(v)
        }
    }
    let mut p = P {
        s: s.as_bytes(),
        i: 0,
    };
    let v = p.expr()?;
    p.skip();
    if p.i != p.s.len() {
        return None;
    }
    Some(v)
}

pub fn apply_eval_to_material(
    material: &mut Material,
    textures: &mut Vec<TextureData>,
    eval: GraphEvalResult,
) {
    material.base_color = eval.base_color;
    material.metallic = eval.metallic;
    material.roughness = eval.roughness;
    material.emissive = eval.emissive;
    material.normal_scale = eval.normal_scale;
    material.occlusion_strength = eval.occlusion_strength;
    material.displacement_scale = eval.displacement_scale;
    material.volume_density = eval.volume_density;

    for t in eval.new_textures {
        textures.push(t);
    }
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
    let order = topo_order(graph);
    let links = input_link_map(graph);
    let mut cache = HashMap::with_capacity(graph.nodes.len() * 3);

    for y in 0..size {
        for x in 0..size {
            let u = (x as f32 + 0.5) / size as f32;
            let v = (y as f32 + 0.5) / size as f32;
            let color = evaluate_graph_point(
                graph, &links, &order, &node_map, textures, material, u, v, &mut cache,
            )
            .base_color;
            let i = ((y * size + x) * 4) as usize;
            rgba[i] = linear_to_srgb8(color[0]);
            rgba[i + 1] = linear_to_srgb8(color[1]);
            rgba[i + 2] = linear_to_srgb8(color[2]);
            rgba[i + 3] = (color[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    TextureData {
        name: format!("Bake_{size}"),
        width: size,
        height: size,
        rgba8: rgba,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link(from: u64, from_socket: u8, to: u64, to_socket: u8) -> ShaderLink {
        ShaderLink {
            from,
            from_socket,
            to,
            to_socket,
        }
    }

    #[test]
    fn noise_defaults_and_socket_values_are_independent() {
        let mut node = ShaderNode::with_kind(1, ShaderNodeKind::Noise, [0.0, 0.0]);
        assert_eq!(node.mode, 2);
        assert_eq!(node.socket_default(2)[0], 5.0);
        assert_eq!(node.socket_default(4)[0], 0.5);
        node.set_socket_default(5, [3.25, 0.0, 0.0, 0.0]);
        node.set_socket_default(4, [0.8, 0.0, 0.0, 0.0]);
        assert_eq!(node.socket_default(5)[0], 3.25);
        assert_eq!(node.socket_default(4)[0], 0.8);
    }

    #[test]
    fn noise_texture_is_perlin_fractal_with_distinct_color_channels() {
        let (color, factor) = noise_texture([0.17, 0.31, 0.53], 0.29, 2, 4.7, 2.5, 0.5, 2.0, 0.0);
        assert!((0.0..=1.0).contains(&factor));
        assert_ne!(color[0], color[1]);
        assert_ne!(color[1], color[2]);
        let nearby = noise_texture([0.1701, 0.31, 0.53], 0.29, 2, 4.7, 2.5, 0.5, 2.0, 0.0).1;
        assert_ne!(factor, nearby);
    }

    #[test]
    fn white_noise_uses_unquantized_seeds_and_independent_color_channels() {
        let (value, color) = white_noise_texture([0.25, 0.5, 0.75], 0.125, 3);
        let neighboring = white_noise_texture([0.2501, 0.5, 0.75], 0.125, 3).0;
        assert!((0.0..=1.0).contains(&value));
        assert_ne!(value, neighboring);
        assert_ne!(color[0], color[1]);
        assert_ne!(color[1], color[2]);
    }

    #[test]
    fn cancelled_graph_evaluation_stops_before_baking() {
        let cancel = AtomicBool::new(true);
        let result = evaluate_graph_ex_cancellable(
            &ShaderGraph::default(),
            &Material::default(),
            &[],
            128,
            false,
            1.0,
            0.0,
            &cancel,
        );
        assert!(result.is_none());
    }

    #[test]
    fn brick_factor_is_one_in_mortar_and_zero_inside_bricks() {
        let mut brick = ShaderNode::with_kind(1, ShaderNodeKind::Brick, [0.0, 0.0]);
        let principled = ShaderNode::with_kind(2, ShaderNodeKind::Principled, [0.0, 0.0]);
        let output = ShaderNode::with_kind(3, ShaderNodeKind::Output, [0.0, 0.0]);
        brick.set_socket_default(4, [1.0, 0.0, 0.0, 0.0]);
        let graph = ShaderGraph {
            nodes: vec![brick, principled, output],
            links: vec![link(1, 1, 2, 13), link(2, 0, 3, 0)],
            next_id: 4,
        };
        let node_map: HashMap<u64, &ShaderNode> =
            graph.nodes.iter().map(|node| (node.id, node)).collect();
        let links = input_link_map(&graph);
        let order = topo_order(&graph);
        let mut cache = HashMap::new();
        let mortar = evaluate_graph_point(
            &graph,
            &links,
            &order,
            &node_map,
            &[],
            &Material::default(),
            0.0,
            0.0,
            &mut cache,
        );
        let brick_center = evaluate_graph_point(
            &graph,
            &links,
            &order,
            &node_map,
            &[],
            &Material::default(),
            0.125,
            0.125,
            &mut cache,
        );
        assert!((mortar.occlusion - 1.0).abs() < 1e-6);
        assert!((brick_center.occlusion - 0.0).abs() < 1e-6);
    }

    #[test]
    fn math_modes_follow_the_editor_operation_order() {
        assert_eq!(math_op(0, 2.0, 3.0), 5.0);
        assert_eq!(math_op(2, 2.0, 3.0), 6.0);
        assert_eq!(math_op(7, 4.0, 0.0), 0.5);
        assert_eq!(math_op(12, 1.0, 2.0), 1.0);
        assert!((math_op(37, 180.0, 0.0) - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn voronoi_bake_uses_the_same_feature_modes_as_graph_evaluation() {
        for mode in 0..=4 {
            let baked = voronoi_ex(0.37, 0.61, 0.8, 0.5, mode);
            let (distance, color, _, _) = voronoi_sample(0.37, 0.61, 0.8, 0.5, mode);
            assert_eq!(baked.0, distance);
            assert_eq!(baked.1, [color[0], color[1], color[2]]);
        }
    }

    #[test]
    fn voronoi_modes_select_distinct_features_and_radius() {
        let f1 = voronoi_sample(0.37, 0.61, 0.8, 0.5, 0);
        let f2 = voronoi_sample(0.37, 0.61, 0.8, 0.5, 1);
        let smooth = voronoi_sample(0.37, 0.61, 0.8, 0.5, 2);
        let edge = voronoi_sample(0.37, 0.61, 0.8, 0.5, 3);
        let sphere = voronoi_sample(0.37, 0.61, 0.8, 0.5, 4);
        assert_ne!(f1.0, f2.0);
        assert_ne!(f1.1, f2.1);
        assert_ne!(f1.2, f2.2);
        assert_ne!(f1.0, smooth.0);
        assert_ne!(f1.0, edge.0);
        assert_ne!(sphere.3, 0.0);
        assert_eq!(f1.3, sphere.3);
    }

    #[test]
    fn graph_bakes_color_and_pbr_channels_per_uv() {
        let noise = ShaderNode::with_kind(1, ShaderNodeKind::Noise, [0.0, 0.0]);
        let mut value = ShaderNode::with_kind(2, ShaderNodeKind::Value, [0.0, 0.0]);
        value.value = 0.5;
        let bump = ShaderNode::with_kind(3, ShaderNodeKind::Bump, [0.0, 0.0]);
        let principled = ShaderNode::with_kind(4, ShaderNodeKind::Principled, [0.0, 0.0]);
        let output = ShaderNode::with_kind(5, ShaderNodeKind::Output, [0.0, 0.0]);
        let graph = ShaderGraph {
            nodes: vec![noise, value, bump, principled, output],
            links: vec![
                link(1, 1, 4, 0),
                link(1, 0, 4, 1),
                link(1, 0, 4, 2),
                link(1, 0, 4, 13),
                link(1, 1, 4, 6),
                link(2, 0, 4, 7),
                link(1, 0, 3, 2),
                link(3, 0, 4, 5),
                link(4, 0, 5, 0),
            ],
            next_id: 6,
        };
        let eval = evaluate_graph_ex(&graph, &Material::default(), &[], 64, false, 1.0, 0.0);
        assert!(eval.base_color_tex.is_some());
        assert!(eval.metallic_roughness_tex.is_some());
        assert!(eval.emissive_tex.is_some());
        assert!(eval.occlusion_tex.is_some());
        assert!(eval.normal_tex.is_some());

        let texture = |name: &str| {
            eval.new_textures
                .iter()
                .find(|texture| texture.name == name)
                .unwrap()
        };
        let base = texture("__sg_BaseColor");
        let first = &base.rgba8[0..4];
        let last = &base.rgba8[(64 * 64 - 1) * 4..64 * 64 * 4];
        assert_ne!(first, last);
        let mr = texture("__sg_MetalRough");
        assert_ne!(mr.rgba8[1], mr.rgba8[(64 * 64 - 1) * 4 + 1]);
        assert_ne!(mr.rgba8[2], mr.rgba8[(64 * 64 - 1) * 4 + 2]);
        let normal = texture("__sg_Normal");
        assert!(normal
            .rgba8
            .chunks_exact(4)
            .any(|pixel| pixel[0] != 128 || pixel[1] != 128));
    }

    #[test]
    fn mapping_rotation_is_applied_after_scale() {
        let rotated = rotate_euler_xyz([1.0, 0.0, 0.0], [0.0, 0.0, std::f32::consts::FRAC_PI_2]);
        assert!(rotated[0].abs() < 1e-6);
        assert!((rotated[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn image_colors_are_decoded_to_linear_before_graph_math() {
        let texture = TextureData {
            name: "gray".into(),
            width: 1,
            height: 1,
            rgba8: vec![128, 128, 128, 255],
        };
        let linear = sample_tex(&texture, 0.5, 0.5);
        assert!((linear[0] - 0.216).abs() < 0.002);
        assert_eq!(sample_tex_data(&texture, 0.5, 0.5)[0], 128.0 / 255.0);
    }

    #[test]
    fn math_to_ramp_bakes_the_selected_operation() {
        let a = ShaderNode::with_kind(1, ShaderNodeKind::Value, [0.0, 0.0]);
        let b = ShaderNode::with_kind(2, ShaderNodeKind::Value, [0.0, 0.0]);
        let mut math = ShaderNode::with_kind(3, ShaderNodeKind::Math, [0.0, 0.0]);
        math.mode = 2;
        let mut ramp = ShaderNode::with_kind(4, ShaderNodeKind::ColorRamp, [0.0, 0.0]);
        ramp.value = 2.0;
        ramp.color = [0.0, 0.0, 0.0, 1.0];
        ramp.props[0..4].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        ramp.props[12] = 0.0;
        ramp.props[13] = 1.0;
        let principled = ShaderNode::with_kind(5, ShaderNodeKind::Principled, [0.0, 0.0]);
        let output = ShaderNode::with_kind(6, ShaderNodeKind::Output, [0.0, 0.0]);
        let mut a = a;
        let mut b = b;
        a.value = 0.5;
        b.value = 0.5;
        let graph = ShaderGraph {
            nodes: vec![a, b, math, ramp, principled, output],
            links: vec![
                link(1, 0, 3, 0),
                link(2, 0, 3, 1),
                link(3, 0, 4, 0),
                link(4, 0, 5, 0),
                link(5, 0, 6, 0),
            ],
            next_id: 7,
        };
        let eval = evaluate_graph_ex(&graph, &Material::default(), &[], 64, false, 1.0, 0.0);
        let baked = eval
            .new_textures
            .iter()
            .find(|texture| texture.name == "__sg_BaseColor")
            .unwrap();
        assert_eq!(baked.rgba8[0], 137);
        assert_eq!(baked.rgba8[1], 137);
        assert_eq!(baked.rgba8[2], 137);
    }

    #[test]
    fn white_noise_dimensions_use_the_correct_seed_inputs() {
        let one_d = white_noise_texture([0.1, 0.2, 0.3], 0.4, 0).0;
        assert_eq!(one_d, white_noise_texture([0.8, 0.7, 0.6], 0.4, 0).0);
        assert_ne!(one_d, white_noise_texture([0.1, 0.2, 0.3], 0.5, 0).0);

        let two_d = white_noise_texture([0.1, 0.2, 0.3], 0.4, 1).0;
        assert_eq!(two_d, white_noise_texture([0.1, 0.2, 0.9], 0.8, 1).0);
        assert_ne!(two_d, white_noise_texture([0.9, 0.2, 0.3], 0.4, 1).0);

        let three_d = white_noise_texture([0.1, 0.2, 0.3], 0.4, 2).0;
        assert_ne!(three_d, white_noise_texture([0.1, 0.2, 0.9], 0.4, 2).0);
        assert_ne!(
            white_noise_texture([0.1, 0.2, 0.3], 0.4, 3).0,
            white_noise_texture([0.1, 0.2, 0.3], 0.5, 3).0
        );
    }

    #[test]
    fn wave_distortion_uses_detail_and_direction_settings() {
        let point = [0.13, 0.27, 0.41];
        let undistorted = wave_value(point, 5.0, 0.0, 2.0, 1.0, 0.5, 0.0, 0, 0, 0, 0);
        let mapped_point = point.map(|component| (component * 5.0 + 0.000001) * 0.999999);
        let expected = 0.5
            + 0.5 * ((mapped_point[0] * 20.0) - std::f32::consts::FRAC_PI_2).sin();
        assert!((undistorted - expected).abs() < 1e-6);
        let distorted = wave_value(point, 5.0, 4.0, 4.0, 2.0, 0.8, 0.0, 0, 0, 0, 0);
        assert_ne!(undistorted, distorted);
        let rings_x = wave_value(point, 5.0, 0.0, 2.0, 1.0, 0.5, 0.0, 1, 0, 0, 0);
        let rings_y = wave_value(point, 5.0, 0.0, 2.0, 1.0, 0.5, 0.0, 1, 0, 1, 0);
        assert_ne!(rings_x, rings_y);
    }

    #[test]
    fn magic_texture_matches_the_reference_formula_and_depth_changes_output() {
        let point = [0.1, 0.2, 0.3];
        let (color, _) = magic_texture(point, 2.0, 1.0, 0);
        let co = point.map(|component| component * 2.0);
        let expected = [
            0.5 - ((co[0] + co[1] + co[2]) * 5.0).sin() * 0.5,
            0.5 - ((-co[0] + co[1] - co[2]) * 5.0).cos() * 0.5,
            0.5 + ((-co[0] - co[1] + co[2]) * 5.0).cos() * 0.5,
            1.0,
        ];
        for component in 0..4 {
            assert!((color[component] - expected[component]).abs() < 1e-6);
        }
        assert_ne!(
            color,
            magic_texture(point, 2.0, 1.0, 5).0,
            "Magic depth must affect the generated color"
        );
    }

    #[test]
    fn brick_offset_and_squash_properties_change_the_pattern() {
        let colors = ([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]);
        let sample = |point, offset, squash| {
            brick_texture(
                point,
                1.0,
                colors.0,
                colors.1,
                [0.0, 0.0, 0.0, 1.0],
                0.08,
                0.0,
                0.0,
                0.5,
                0.25,
                offset,
                2,
                squash,
                2,
            )
        };
        assert_ne!(sample([0.02, 0.1, 0.0], 0.0, 1.0).1, sample([0.02, 0.1, 0.0], 0.5, 1.0).1);
        assert_ne!(sample([0.3, 0.1, 0.0], 0.0, 1.0).1, sample([0.3, 0.1, 0.0], 0.0, 0.5).1);
    }
}
