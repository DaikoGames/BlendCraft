//! Serializable material shader graph and CPU evaluation into PBR slots / textures.
//! Node set aligned with Blender Shader Editor categories (Input, Output, Shader, Texture, Color, Vector/Utilities).
use crate::material::{Material, TextureData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
            Principled | Diffuse | Glossy | Emission | AddShader | MixShader | Background | Glass
            | Hair | Holdout | Metallic | PrincipledHair | PrincipledVolume | RayPortal | Refraction
            | Specular | Subsurface | Toon | Translucent | Transparent | Sheen | VolumeAbsorption
            | VolumeScatter | VolumeCoefficients | LightAccumulation => "Shader",
            ImageTexture | Noise | Voronoi | Brick | Checker | Environment | Gabor | Gradient | Ies
            | Magic | Sky | Wave | WhiteNoise => "Texture",
            ColorRamp | Mix | Blackbody | BrightContrast | Gamma | HueSatVal | InvertColor
            | LightFalloff | RgbCurves | Wavelength | CombineColor | SeparateColor | RgbToBw
            | ShaderToRgb => "Color",
            Mapping | Bump | NormalMap | CombineXyz | SeparateXyz | VectorMath | VectorRotate
            | VectorTransform | VectorCurves | Normal | Displacement | VectorDisplacement | Math
            | MapRange | Clamp | FloatCurve | Switch | IndexSwitch | MenuSwitch | Script | Group => {
                "Vector"
            }
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
                SocketDef { name: "Surface", ty: Shader, input: true },
                SocketDef { name: "Volume", ty: Shader, input: true },
                SocketDef { name: "Displacement", ty: Vector, input: true },
                SocketDef { name: "Thickness", ty: Float, input: true },
            ],
            LightOutput | WorldOutput => &[
                SocketDef { name: "Surface", ty: Shader, input: true },
                SocketDef { name: "Volume", ty: Shader, input: true },
            ],
            AovOutput => &[
                SocketDef { name: "Color", ty: Color, input: true },
                SocketDef { name: "Value", ty: Float, input: true },
            ],
            Principled => &[
                SocketDef { name: "Base Color", ty: Color, input: true },
                SocketDef { name: "Metallic", ty: Float, input: true },
                SocketDef { name: "Roughness", ty: Float, input: true },
                SocketDef { name: "IOR", ty: Float, input: true },
                SocketDef { name: "Alpha", ty: Float, input: true },
                SocketDef { name: "Normal", ty: Vector, input: true },
                SocketDef { name: "Emission Color", ty: Color, input: true },
                SocketDef { name: "Emission Strength", ty: Float, input: true },
                SocketDef { name: "Coat Weight", ty: Float, input: true },
                SocketDef { name: "Sheen Weight", ty: Float, input: true },
                SocketDef { name: "Subsurface Weight", ty: Float, input: true },
                SocketDef { name: "Transmission Weight", ty: Float, input: true },
                SocketDef { name: "Specular IOR Level", ty: Float, input: true },
                SocketDef { name: "AO", ty: Float, input: true },
            ],
            Diffuse => &[
                SocketDef { name: "Color", ty: Color, input: true },
                SocketDef { name: "Roughness", ty: Float, input: true },
                SocketDef { name: "Normal", ty: Vector, input: true },
            ],
            Glossy => &[
                SocketDef { name: "Color", ty: Color, input: true },
                SocketDef { name: "Roughness", ty: Float, input: true },
                SocketDef { name: "Anisotropy", ty: Float, input: true },
                SocketDef { name: "Rotation", ty: Float, input: true },
                SocketDef { name: "Normal", ty: Vector, input: true },
            ],
            Emission | Background => &[
                SocketDef { name: "Color", ty: Color, input: true },
                SocketDef { name: "Strength", ty: Float, input: true },
            ],
            AddShader => &[
                SocketDef { name: "Shader", ty: Shader, input: true },
                SocketDef { name: "Shader", ty: Shader, input: true },
            ],
            MixShader => &[
                SocketDef { name: "Fac", ty: Float, input: true },
                SocketDef { name: "Shader", ty: Shader, input: true },
                SocketDef { name: "Shader", ty: Shader, input: true },
            ],
            Glass | Refraction => &[
                SocketDef { name: "Color", ty: Color, input: true },
                SocketDef { name: "Roughness", ty: Float, input: true },
                SocketDef { name: "IOR", ty: Float, input: true },
                SocketDef { name: "Normal", ty: Vector, input: true },
            ],
            Transparent | Translucent | Sheen | Holdout => &[
                SocketDef { name: "Color", ty: Color, input: true },
                SocketDef { name: "Normal", ty: Vector, input: true },
            ],
            VolumeAbsorption | VolumeScatter | VolumeCoefficients | PrincipledVolume => &[
                SocketDef { name: "Color", ty: Color, input: true },
                SocketDef { name: "Density", ty: Float, input: true },
                SocketDef { name: "Anisotropy", ty: Float, input: true },
            ],
            ImageTexture => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
            ],
            Noise => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
                SocketDef { name: "W", ty: Float, input: true },
                SocketDef { name: "Scale", ty: Float, input: true },
                SocketDef { name: "Detail", ty: Float, input: true },
                SocketDef { name: "Roughness", ty: Float, input: true },
                SocketDef { name: "Lacunarity", ty: Float, input: true },
                SocketDef { name: "Distortion", ty: Float, input: true },
            ],
            Voronoi => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
                SocketDef { name: "W", ty: Float, input: true },
                SocketDef { name: "Scale", ty: Float, input: true },
                SocketDef { name: "Detail", ty: Float, input: true },
                SocketDef { name: "Roughness", ty: Float, input: true },
                SocketDef { name: "Lacunarity", ty: Float, input: true },
                SocketDef { name: "Smoothness", ty: Float, input: true },
                SocketDef { name: "Randomness", ty: Float, input: true },
            ],
            Brick => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
                SocketDef { name: "Color1", ty: Color, input: true },
                SocketDef { name: "Color2", ty: Color, input: true },
                SocketDef { name: "Mortar", ty: Color, input: true },
                SocketDef { name: "Scale", ty: Float, input: true },
                SocketDef { name: "Mortar Size", ty: Float, input: true },
                SocketDef { name: "Mortar Smooth", ty: Float, input: true },
                SocketDef { name: "Bias", ty: Float, input: true },
                SocketDef { name: "Brick Width", ty: Float, input: true },
                SocketDef { name: "Row Height", ty: Float, input: true },
            ],
            Checker => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
                SocketDef { name: "Color1", ty: Color, input: true },
                SocketDef { name: "Color2", ty: Color, input: true },
                SocketDef { name: "Scale", ty: Float, input: true },
            ],
            Environment => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
            ],
            Gabor => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
                SocketDef { name: "Scale", ty: Float, input: true },
                SocketDef { name: "Frequency", ty: Float, input: true },
                SocketDef { name: "Anisotropy", ty: Float, input: true },
                SocketDef { name: "Orientation", ty: Float, input: true },
            ],
            Gradient => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
            ],
            Ies => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
                SocketDef { name: "Strength", ty: Float, input: true },
            ],
            Magic => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
                SocketDef { name: "Scale", ty: Float, input: true },
                SocketDef { name: "Distortion", ty: Float, input: true },
            ],
            Sky => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
            ],
            Wave => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
                SocketDef { name: "Scale", ty: Float, input: true },
                SocketDef { name: "Distortion", ty: Float, input: true },
                SocketDef { name: "Detail", ty: Float, input: true },
                SocketDef { name: "Detail Scale", ty: Float, input: true },
                SocketDef { name: "Detail Roughness", ty: Float, input: true },
                SocketDef { name: "Phase Offset", ty: Float, input: true },
            ],
            WhiteNoise => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
                SocketDef { name: "W", ty: Float, input: true },
            ],
            ColorRamp => &[
                SocketDef { name: "Fac", ty: Float, input: true },
            ],
            Mix => &[
                SocketDef { name: "Fac", ty: Float, input: true },
                SocketDef { name: "A", ty: Color, input: true },
                SocketDef { name: "B", ty: Color, input: true },
            ],
            Blackbody | Wavelength => &[
                SocketDef { name: "Temperature", ty: Float, input: true },
            ],
            BrightContrast => &[
                SocketDef { name: "Color", ty: Color, input: true },
                SocketDef { name: "Bright", ty: Float, input: true },
                SocketDef { name: "Contrast", ty: Float, input: true },
            ],
            Gamma => &[
                SocketDef { name: "Color", ty: Color, input: true },
                SocketDef { name: "Gamma", ty: Float, input: true },
            ],
            HueSatVal => &[
                SocketDef { name: "Color", ty: Color, input: true },
                SocketDef { name: "Hue", ty: Float, input: true },
                SocketDef { name: "Saturation", ty: Float, input: true },
                SocketDef { name: "Value", ty: Float, input: true },
                SocketDef { name: "Fac", ty: Float, input: true },
            ],
            InvertColor => &[
                SocketDef { name: "Fac", ty: Float, input: true },
                SocketDef { name: "Color", ty: Color, input: true },
            ],
            LightFalloff => &[
                SocketDef { name: "Strength", ty: Float, input: true },
                SocketDef { name: "Smooth", ty: Float, input: true },
            ],
            RgbCurves | RgbToBw | ShaderToRgb => &[
                SocketDef { name: "Color", ty: Color, input: true },
                SocketDef { name: "Fac", ty: Float, input: true },
            ],
            CombineColor | CombineXyz => &[
                SocketDef { name: "R/X", ty: Float, input: true },
                SocketDef { name: "G/Y", ty: Float, input: true },
                SocketDef { name: "B/Z", ty: Float, input: true },
            ],
            SeparateColor | SeparateXyz => &[
                SocketDef { name: "Color/Vector", ty: Color, input: true },
            ],
            Mapping => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
                SocketDef { name: "Location", ty: Vector, input: true },
                SocketDef { name: "Rotation", ty: Vector, input: true },
                SocketDef { name: "Scale", ty: Vector, input: true },
            ],
            Bump => &[
                SocketDef { name: "Strength", ty: Float, input: true },
                SocketDef { name: "Distance", ty: Float, input: true },
                SocketDef { name: "Height", ty: Float, input: true },
                SocketDef { name: "Normal", ty: Vector, input: true },
            ],
            NormalMap => &[
                SocketDef { name: "Strength", ty: Float, input: true },
                SocketDef { name: "Color", ty: Color, input: true },
            ],
            VectorMath => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
                SocketDef { name: "Vector", ty: Vector, input: true },
                SocketDef { name: "Scale", ty: Float, input: true },
            ],
            VectorRotate => &[
                SocketDef { name: "Vector", ty: Vector, input: true },
                SocketDef { name: "Center", ty: Vector, input: true },
                SocketDef { name: "Axis", ty: Vector, input: true },
                SocketDef { name: "Angle", ty: Float, input: true },
            ],
            Math => &[
                SocketDef { name: "Value", ty: Float, input: true },
                SocketDef { name: "Value", ty: Float, input: true },
            ],
            MapRange => &[
                SocketDef { name: "Value", ty: Float, input: true },
                SocketDef { name: "From Min", ty: Float, input: true },
                SocketDef { name: "From Max", ty: Float, input: true },
                SocketDef { name: "To Min", ty: Float, input: true },
                SocketDef { name: "To Max", ty: Float, input: true },
            ],
            Clamp => &[
                SocketDef { name: "Value", ty: Float, input: true },
                SocketDef { name: "Min", ty: Float, input: true },
                SocketDef { name: "Max", ty: Float, input: true },
            ],
            Switch | IndexSwitch | MenuSwitch => &[
                SocketDef { name: "Switch", ty: Float, input: true },
                SocketDef { name: "False", ty: Color, input: true },
                SocketDef { name: "True", ty: Color, input: true },
            ],
            FloatCurve | VectorCurves | VectorTransform | Displacement | VectorDisplacement | Normal => &[
                SocketDef { name: "Value", ty: Float, input: true },
                SocketDef { name: "Vector", ty: Vector, input: true },
            ],
            Script => &[
                SocketDef { name: "a", ty: Float, input: true },
                SocketDef { name: "b", ty: Float, input: true },
            ],
            Fresnel => &[
                SocketDef { name: "IOR", ty: Float, input: true },
                SocketDef { name: "Normal", ty: Vector, input: true },
            ],
            LayerWeight => &[
                SocketDef { name: "Blend", ty: Float, input: true },
                SocketDef { name: "Normal", ty: Vector, input: true },
            ],
            AmbientOcclusion => &[
                SocketDef { name: "Color", ty: Color, input: true },
                SocketDef { name: "Distance", ty: Float, input: true },
                SocketDef { name: "Normal", ty: Vector, input: true },
            ],
            Bevel => &[
                SocketDef { name: "Radius", ty: Float, input: true },
                SocketDef { name: "Normal", ty: Vector, input: true },
            ],
            Raycast | ShadowRaycast => &[
                SocketDef { name: "Position", ty: Vector, input: true },
                SocketDef { name: "Direction", ty: Vector, input: true },
                SocketDef { name: "Length", ty: Float, input: true },
            ],
            Wireframe => &[
                SocketDef { name: "Size", ty: Float, input: true },
            ],
            _ => &[],
        }
    }

    pub fn outputs(self) -> &'static [SocketDef] {
        use ShaderNodeKind::*;
        use SocketType::*;
        match self {
            Output | LightOutput | WorldOutput | AovOutput => &[],
            Principled | Diffuse | Glossy | Emission | AddShader | MixShader | Background | Glass
            | Hair | Holdout | Metallic | PrincipledHair | PrincipledVolume | RayPortal | Refraction
            | Specular | Subsurface | Toon | Translucent | Transparent | Sheen | VolumeAbsorption
            | VolumeScatter | VolumeCoefficients | LightAccumulation => &[
                SocketDef { name: "BSDF", ty: Shader, input: false },
            ],
            ImageTexture | Environment => &[
                SocketDef { name: "Color", ty: Color, input: false },
                SocketDef { name: "Alpha", ty: Float, input: false },
            ],
            Noise | Wave | Magic | Gradient | Gabor | Ies | WhiteNoise => &[
                SocketDef { name: "Fac", ty: Float, input: false },
                SocketDef { name: "Color", ty: Color, input: false },
            ],
            Voronoi => &[
                SocketDef { name: "Distance", ty: Float, input: false },
                SocketDef { name: "Color", ty: Color, input: false },
                SocketDef { name: "Position", ty: Vector, input: false },
                SocketDef { name: "W", ty: Float, input: false },
                SocketDef { name: "Radius", ty: Float, input: false },
            ],
            Brick | Checker => &[
                SocketDef { name: "Color", ty: Color, input: false },
                SocketDef { name: "Fac", ty: Float, input: false },
            ],
            Sky => &[
                SocketDef { name: "Color", ty: Color, input: false },
            ],
            ColorRamp | Mix | Blackbody | BrightContrast | Gamma
            | HueSatVal | InvertColor | RgbCurves | Wavelength | Rgb | ColorAttribute => &[
                SocketDef { name: "Color", ty: Color, input: false },
                SocketDef { name: "Alpha", ty: Float, input: false },
            ],
            RgbToBw | Value | Math | MapRange | Clamp | FloatCurve | Fresnel | Wireframe
            | AmbientOcclusion | SceneTime => &[
                SocketDef { name: "Value", ty: Float, input: false },
            ],
            SeparateColor | SeparateXyz => &[
                SocketDef { name: "R/X", ty: Float, input: false },
                SocketDef { name: "G/Y", ty: Float, input: false },
                SocketDef { name: "B/Z", ty: Float, input: false },
            ],
            CombineColor | CombineXyz | Mapping | VectorMath | VectorRotate | VectorTransform
            | VectorCurves | Normal | Displacement | VectorDisplacement | TexCoord | UvMap
            | Tangent | Geometry | Bevel | NormalMap | Bump => &[
                SocketDef { name: "Vector", ty: Vector, input: false },
            ],
            LayerWeight => &[
                SocketDef { name: "Fresnel", ty: Float, input: false },
                SocketDef { name: "Facing", ty: Float, input: false },
            ],
            CameraData => &[
                SocketDef { name: "View Vector", ty: Vector, input: false },
                SocketDef { name: "View Z Depth", ty: Float, input: false },
                SocketDef { name: "View Distance", ty: Float, input: false },
            ],
            LightPath => &[
                SocketDef { name: "Is Camera Ray", ty: Float, input: false },
                SocketDef { name: "Is Shadow Ray", ty: Float, input: false },
                SocketDef { name: "Is Diffuse Ray", ty: Float, input: false },
                SocketDef { name: "Is Glossy Ray", ty: Float, input: false },
                SocketDef { name: "Ray Depth", ty: Float, input: false },
            ],
            ObjectInfo => &[
                SocketDef { name: "Location", ty: Vector, input: false },
                SocketDef { name: "Color", ty: Color, input: false },
                SocketDef { name: "Object Index", ty: Float, input: false },
                SocketDef { name: "Material Index", ty: Float, input: false },
                SocketDef { name: "Random", ty: Float, input: false },
            ],
            ShaderToRgb => &[
                SocketDef { name: "Color", ty: Color, input: false },
                SocketDef { name: "Alpha", ty: Float, input: false },
            ],
            LightFalloff => &[
                SocketDef { name: "Quadratic", ty: Float, input: false },
                SocketDef { name: "Linear", ty: Float, input: false },
                SocketDef { name: "Constant", ty: Float, input: false },
            ],
            Switch | IndexSwitch | MenuSwitch | Group | Script => &[
                SocketDef { name: "Output", ty: Color, input: false },
            ],
            _ => &[
                SocketDef { name: "Value", ty: Float, input: false },
            ],
        }
    }

    pub fn is_input_extra(self) -> bool {
        self.category() == "Input"
            && !matches!(self, Self::Rgb | Self::Value | Self::TexCoord)
    }

    pub fn all_kinds() -> &'static [ShaderNodeKind] {
        use ShaderNodeKind::*;
        &[
            Output, LightOutput, WorldOutput, AovOutput,
            Principled, Diffuse, Glossy, Emission, AddShader, MixShader, Background, Glass, Hair,
            Holdout, Metallic, PrincipledHair, PrincipledVolume, RayPortal, Refraction, Specular,
            Subsurface, Toon, Translucent, Transparent, Sheen, VolumeAbsorption, VolumeScatter,
            VolumeCoefficients, LightAccumulation,
            ImageTexture, Noise, Voronoi, Brick, Checker, Environment, Gabor, Gradient, Ies, Magic,
            Sky, Wave, WhiteNoise,
            ColorRamp, Mix, Blackbody, BrightContrast, Gamma, HueSatVal, InvertColor, LightFalloff,
            RgbCurves, Wavelength, CombineColor, SeparateColor, RgbToBw, ShaderToRgb,
            Mapping, Bump, NormalMap, CombineXyz, SeparateXyz, VectorMath, VectorRotate,
            VectorTransform, VectorCurves, Normal, Displacement, VectorDisplacement, Math, MapRange,
            Clamp, FloatCurve, Switch, IndexSwitch, MenuSwitch, Script, Group,
            Rgb, Value, TexCoord, UvMap, AmbientOcclusion, Attribute, Bevel, CameraData, Fresnel,
            Geometry, CurvesInfo, LayerWeight, LightEvaluation, LightInfo, LightPath, ObjectInfo,
            ParticleInfo, PointInfo, Raycast, SceneTime, ShadowRaycast, Tangent, ColorAttribute,
            VolumeInfo, Wireframe,
        ]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
            mode: 0,
            name_prop: String::new(),
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
            if node.props[3].abs() < 1e-8 { 1.0 } else { node.props[3] },
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
            if node.props[7].abs() < 1e-8 { 1.0 } else { node.props[7] },
        ]);
        poss.push(node.props[14].clamp(0.0, 1.0));
    }
    if n >= 4 {
        cols.push([
            node.props[8],
            node.props[9],
            node.props[10],
            if node.props[11].abs() < 1e-8 { 1.0 } else { node.props[11] },
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
    order.sort_by(|&i, &j| poss[i].partial_cmp(&poss[j]).unwrap_or(std::cmp::Ordering::Equal));
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
                2 => 0.0, // Constant: hold left color of this segment
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

/// Legacy black→white helper.
fn color_ramp(t: f32) -> [f32; 4] {
    color_ramp_eval_stops(
        t,
        &[[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        &[0.0, 1.0],
        0,
    )
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

pub fn evaluate_graph_ex(
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
        let Some(node) = node_map.get(&id) else { continue };
        let ins = node.kind.inputs();
        let outs = node.kind.outputs();
        // Helper: linked input or default from node.value/color/props
        let inp = |sock: u8, default: SocketVal| -> SocketVal {
            find_input(graph, id, sock, &cache).unwrap_or(default)
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
                let scale = inp(3, SocketVal::Vector([
                    if node.value.abs() < 1e-6 { 1.0 } else { node.value },
                    if node.props[0].abs() < 1e-6 { 1.0 } else { node.props[0] },
                    if node.props[1].abs() < 1e-6 { 1.0 } else { node.props[1] },
                ])).as_vector();
                cache.insert((id, 0), SocketVal::Vector([
                    (v[0] + loc[0]) * scale[0],
                    (v[1] + loc[1]) * scale[1],
                    (v[2] + loc[2]) * scale[2],
                ]));
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
                    let (col, fac) = proc_sample(node.kind, node.mode, node.value, &node.props, node.color, 0.37, 0.61);
                    cache.insert((id, 0), SocketVal::Scalar(fac));
                    cache.insert((id, 1), SocketVal::Color(col));
                    continue;
                }
                let scale = {
                    let s = inp(2, SocketVal::Scalar(if node.value.abs() < 1e-4 { 5.0 } else { node.value })).as_scalar();
                    if s.abs() < 1e-4 { 5.0 } else { s }
                };
                let detail = inp(3, SocketVal::Scalar(if node.props[0] > 0.0 { node.props[0] } else { 5.0 })).as_scalar().clamp(0.0, 15.0);
                let roughness = inp(4, SocketVal::Scalar(if node.props[1] > 0.0 { node.props[1] } else { 0.5 })).as_scalar().clamp(0.0, 1.0);
                let lacunarity = inp(5, SocketVal::Scalar(if node.props[2] > 0.0 { node.props[2] } else { 2.0 })).as_scalar().max(1.0);
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
                        let n = if max_amp > 1e-6 { (sum / max_amp) * 0.5 + 0.5 } else { 0.5 };
                        let i = ((y * size + x) * 4) as usize;
                        let b = (n.clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i] = b; rgba[i+1] = b; rgba[i+2] = b; rgba[i+3] = 255;
                    }
                }
                let tex = TextureData { name: format!("Noise_{id}"), width: size, height: size, rgba8: rgba };
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
                    let (col, fac) = proc_sample(node.kind, node.mode, node.value, &node.props, node.color, 0.37, 0.61);
                    cache.insert((id, 0), SocketVal::Scalar(fac));
                    cache.insert((id, 1), SocketVal::Color(col));
                    cache.insert((id, 2), SocketVal::Vector([0.37, 0.61, 0.0]));
                    cache.insert((id, 3), SocketVal::Scalar(0.0));
                    cache.insert((id, 4), SocketVal::Scalar(fac));
                    continue;
                }
                let scale = {
                    let s = inp(2, SocketVal::Scalar(if node.value.abs() < 1e-4 { 5.0 } else { node.value })).as_scalar();
                    if s.abs() < 1e-4 { 5.0 } else { s }
                };
                let randomness = inp(7, SocketVal::Scalar(if node.props[0] > 0.0 { node.props[0] } else { 1.0 })).as_scalar().clamp(0.0, 1.0);
                let size = tex_size.max(64).min(512);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                let mut mid_d = 0.5f32;
                let mut mid_pos = [0.5f32, 0.5, 0.0];
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32;
                        let v = y as f32 / size as f32;
                        // Feature mode in node.mode: 0=F1, 1=F2, 2=Distance to Edge
                        let (d, cell_col) = voronoi_ex(u * scale, v * scale, randomness, node.mode);
                        let i = ((y * size + x) * 4) as usize;
                        let b = (d.clamp(0.0, 1.0) * 255.0) as u8;
                        let c0 = (cell_col[0] * 255.0) as u8;
                        let c1 = (cell_col[1] * 255.0) as u8;
                        let c2 = (cell_col[2] * 255.0) as u8;
                        // Store distance as grayscale for Fac; color from cell id
                        rgba[i] = c0; rgba[i+1] = c1; rgba[i+2] = c2; rgba[i+3] = b;
                        if x == size / 2 && y == size / 2 {
                            mid_d = d;
                            mid_pos = [u, v, 0.0];
                        }
                    }
                }
                let tex = TextureData { name: format!("Voronoi_{id}"), width: size, height: size, rgba8: rgba };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Scalar(mid_d));
                cache.insert((id, 1), SocketVal::Color([mid[0], mid[1], mid[2], 1.0]));
                cache.insert((id, 2), SocketVal::Vector(mid_pos));
                cache.insert((id, 3), SocketVal::Scalar(0.0));
                cache.insert((id, 4), SocketVal::Scalar(mid_d));
            }
            ShaderNodeKind::Brick => {
                if live {
                    let (col, fac) = proc_sample(node.kind, node.mode, node.value, &node.props, node.color, 0.37, 0.61);
                    cache.insert((id, 0), SocketVal::Color(col));
                    cache.insert((id, 1), SocketVal::Scalar(fac));
                    continue;
                }
                let c1 = inp(1, SocketVal::Color([0.6, 0.25, 0.15, 1.0])).as_color();
                let c2 = inp(2, SocketVal::Color([0.55, 0.22, 0.12, 1.0])).as_color();
                let mortar = inp(3, SocketVal::Color([0.35, 0.35, 0.35, 1.0])).as_color();
                let scale = inp(4, SocketVal::Scalar(if node.value.abs() < 1e-4 { 5.0 } else { node.value })).as_scalar().max(0.01);
                let mortar_size = inp(5, SocketVal::Scalar(if node.props[0] > 0.0 { node.props[0] } else { 0.08 })).as_scalar().clamp(0.0, 0.5);
                let brick_w = inp(8, SocketVal::Scalar(if node.props[1] > 0.0 { node.props[1] } else { 0.5 })).as_scalar().max(0.05);
                let row_h = inp(9, SocketVal::Scalar(if node.props[2] > 0.0 { node.props[2] } else { 0.25 })).as_scalar().max(0.05);
                let size = tex_size.max(64).min(512);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32 * scale;
                        let v = y as f32 / size as f32 * scale;
                        let row = (v / row_h).floor();
                        let offset = if (row as i32) % 2 == 0 { 0.0 } else { brick_w * 0.5 };
                        let bx = (u + offset) / brick_w;
                        let fx = bx.fract();
                        let fy = (v / row_h).fract();
                        let in_mortar = fx < mortar_size || fx > 1.0 - mortar_size || fy < mortar_size || fy > 1.0 - mortar_size;
                        let brick_idx = (bx.floor() as i32).wrapping_add(row as i32 * 17);
                        let use_c2 = brick_idx % 2 == 0;
                        let col = if in_mortar { mortar } else if use_c2 { c2 } else { c1 };
                        let fac = if in_mortar { 0.0 } else { 1.0 };
                        let i = ((y * size + x) * 4) as usize;
                        rgba[i] = (col[0].clamp(0.0,1.0)*255.0) as u8;
                        rgba[i+1] = (col[1].clamp(0.0,1.0)*255.0) as u8;
                        rgba[i+2] = (col[2].clamp(0.0,1.0)*255.0) as u8;
                        rgba[i+3] = (fac * 255.0) as u8;
                    }
                }
                let tex = TextureData { name: format!("Brick_{id}"), width: size, height: size, rgba8: rgba };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Color([mid[0], mid[1], mid[2], 1.0]));
                cache.insert((id, 1), SocketVal::Scalar(mid[3]));
            }
            ShaderNodeKind::Checker => {
                if live {
                    let (col, fac) = proc_sample(node.kind, node.mode, node.value, &node.props, node.color, 0.37, 0.61);
                    cache.insert((id, 0), SocketVal::Color(col));
                    cache.insert((id, 1), SocketVal::Scalar(fac));
                    continue;
                }
                let c1 = inp(1, SocketVal::Color([1.0, 1.0, 1.0, 1.0])).as_color();
                let c2 = inp(2, SocketVal::Color([0.0, 0.0, 0.0, 1.0])).as_color();
                let scale = inp(3, SocketVal::Scalar(if node.value.abs() < 1e-4 { 5.0 } else { node.value })).as_scalar().max(0.01);
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
                        rgba[i] = (col[0].clamp(0.0,1.0)*255.0) as u8;
                        rgba[i+1] = (col[1].clamp(0.0,1.0)*255.0) as u8;
                        rgba[i+2] = (col[2].clamp(0.0,1.0)*255.0) as u8;
                        rgba[i+3] = 255;
                    }
                }
                let tex = TextureData { name: format!("Checker_{id}"), width: size, height: size, rgba8: rgba };
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
                    let (col, fac) = proc_sample(node.kind, node.mode, node.value, &node.props, node.color, 0.37, 0.61);
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
                            2 => { // easing
                                let x = u.clamp(0.0, 1.0);
                                x * x * (3.0 - 2.0 * x)
                            }
                            3 => ((u + v) * 0.5).clamp(0.0, 1.0), // diagonal
                            4 => {
                                let dx = u - 0.5;
                                let dy = v - 0.5;
                                (1.0 - (dx * dx + dy * dy).sqrt() * 2.0).clamp(0.0, 1.0)
                            }
                            5 => { // radial
                                let a = (v - 0.5).atan2(u - 0.5);
                                (a / (2.0 * std::f32::consts::PI) + 0.5).fract()
                            }
                            _ => u.clamp(0.0, 1.0), // linear
                        };
                        let i = ((y * size + x) * 4) as usize;
                        let b = (t * 255.0) as u8;
                        rgba[i] = b; rgba[i+1] = b; rgba[i+2] = b; rgba[i+3] = 255;
                    }
                }
                let tex = TextureData { name: format!("Gradient_{id}"), width: size, height: size, rgba8: rgba };
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
                    let (col, fac) = proc_sample(node.kind, node.mode, node.value, &node.props, node.color, 0.37, 0.61);
                    cache.insert((id, 0), SocketVal::Scalar(fac));
                    cache.insert((id, 1), SocketVal::Color(col));
                    continue;
                }
                let scale = inp(1, SocketVal::Scalar(if node.value.abs() < 1e-4 { 5.0 } else { node.value })).as_scalar().max(0.01);
                let distortion = inp(2, SocketVal::Scalar(node.props[0])).as_scalar();
                let detail = inp(3, SocketVal::Scalar(if node.props[1] > 0.0 { node.props[1] } else { 2.0 })).as_scalar();
                let phase = inp(6, SocketVal::Scalar(node.props[2])).as_scalar();
                // mode: 0 Bands, 1 Rings; props mode profile: 0 Sine, 1 Saw, 2 Triangle via node.mode high nibble - use mode % 3 for profile, mode / 3 for bands/rings
                let bands = node.mode % 2 == 0;
                let profile = (node.mode / 2) % 3;
                let size = tex_size.max(64).min(512);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32;
                        let v = y as f32 / size as f32;
                        let mut p = if bands { u * scale } else {
                            let dx = u - 0.5;
                            let dy = v - 0.5;
                            (dx * dx + dy * dy).sqrt() * scale
                        };
                        p += phase + distortion * noise2(u * scale, v * scale) * detail * 0.1;
                        let t = match profile {
                            1 => p.fract(), // saw
                            2 => {
                                let f = p.fract();
                                if f < 0.5 { f * 2.0 } else { 2.0 - f * 2.0 }
                            }
                            _ => (p * std::f32::consts::TAU).sin() * 0.5 + 0.5, // sine
                        };
                        let i = ((y * size + x) * 4) as usize;
                        let b = (t.clamp(0.0, 1.0) * 255.0) as u8;
                        rgba[i] = b; rgba[i+1] = b; rgba[i+2] = b; rgba[i+3] = 255;
                    }
                }
                let tex = TextureData { name: format!("Wave_{id}"), width: size, height: size, rgba8: rgba };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Scalar(mid[0]));
                cache.insert((id, 1), SocketVal::Color(mid));
            }
            ShaderNodeKind::Magic => {
                if live {
                    let (col, fac) = proc_sample(node.kind, node.mode, node.value, &node.props, node.color, 0.37, 0.61);
                    cache.insert((id, 0), SocketVal::Scalar(fac));
                    cache.insert((id, 1), SocketVal::Color(col));
                    continue;
                }
                let scale = inp(1, SocketVal::Scalar(if node.value.abs() < 1e-4 { 5.0 } else { node.value })).as_scalar().max(0.01);
                let distortion = inp(2, SocketVal::Scalar(if node.props[0].abs() < 1e-6 { 1.0 } else { node.props[0] })).as_scalar();
                let depth = (node.mode as i32 + 2).clamp(2, 10);
                let size = tex_size.max(64).min(512);
                let mut rgba = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let u = x as f32 / size as f32 * scale;
                        let v = y as f32 / size as f32 * scale;
                        let mut px = u;
                        let mut py = v;
                        for _ in 0..depth {
                            px = (px + distortion * py.sin()).sin();
                            py = (py + distortion * px.cos()).cos();
                        }
                        let r = (px * 0.5 + 0.5).clamp(0.0, 1.0);
                        let g = (py * 0.5 + 0.5).clamp(0.0, 1.0);
                        let b = ((px + py) * 0.25 + 0.5).clamp(0.0, 1.0);
                        let i = ((y * size + x) * 4) as usize;
                        rgba[i] = (r * 255.0) as u8;
                        rgba[i+1] = (g * 255.0) as u8;
                        rgba[i+2] = (b * 255.0) as u8;
                        rgba[i+3] = 255;
                    }
                }
                let tex = TextureData { name: format!("Magic_{id}"), width: size, height: size, rgba8: rgba };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Scalar((mid[0]+mid[1]+mid[2])/3.0));
                cache.insert((id, 1), SocketVal::Color(mid));
            }
            ShaderNodeKind::Gabor => {
                if live {
                    let (col, fac) = proc_sample(node.kind, node.mode, node.value, &node.props, node.color, 0.37, 0.61);
                    cache.insert((id, 0), SocketVal::Scalar(fac));
                    cache.insert((id, 1), SocketVal::Color(col));
                    continue;
                }
                let scale = inp(1, SocketVal::Scalar(5.0)).as_scalar().max(0.01);
                let frequency = inp(2, SocketVal::Scalar(if node.props[0] > 0.0 { node.props[0] } else { 2.0 })).as_scalar();
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
                        rgba[i] = b; rgba[i+1] = b; rgba[i+2] = b; rgba[i+3] = 255;
                    }
                }
                let tex = TextureData { name: format!("Gabor_{id}"), width: size, height: size, rgba8: rgba };
                let new_idx = textures.len() + result.new_textures.len();
                result.new_textures.push(tex);
                proc_tex_index.insert(id, new_idx);
                let mid = sample_tex(result.new_textures.last().unwrap(), 0.5, 0.5);
                cache.insert((id, 0), SocketVal::Scalar(mid[0]));
                cache.insert((id, 1), SocketVal::Color(mid));
            }
            ShaderNodeKind::WhiteNoise => {
                if live {
                    let (col, fac) = proc_sample(node.kind, node.mode, node.value, &node.props, node.color, 0.37, 0.61);
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
                        // 2D or 3D hash depending on mode (0=2D, 1=3D uses W)
                        let w = inp(1, SocketVal::Scalar(0.0)).as_scalar();
                        let n = if node.mode == 0 {
                            hash2(u * 1000.0, v * 1000.0)
                        } else {
                            hash2(hash2(u * 1000.0, v * 1000.0) * 1000.0, w * 1000.0)
                        };
                        let i = ((y * size + x) * 4) as usize;
                        let b = (n * 255.0) as u8;
                        rgba[i] = b; rgba[i+1] = b; rgba[i+2] = b; rgba[i+3] = 255;
                    }
                }
                let tex = TextureData { name: format!("WhiteNoise_{id}"), width: size, height: size, rgba8: rgba };
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
                        rgba[i] = (r.clamp(0.0,1.0)*255.0) as u8;
                        rgba[i+1] = (g.clamp(0.0,1.0)*255.0) as u8;
                        rgba[i+2] = (b.clamp(0.0,1.0)*255.0) as u8;
                        rgba[i+3] = 255;
                    }
                }
                if !live {
                    let tex = TextureData { name: format!("Sky_{id}"), width: size, height: size, rgba8: rgba.clone() };
                    let new_idx = textures.len() + result.new_textures.len();
                    result.new_textures.push(tex);
                    proc_tex_index.insert(id, new_idx);
                }
                cache.insert((id, 0), SocketVal::Color([0.55, 0.65, 0.9, 1.0]));
            }
            ShaderNodeKind::Ies => {
                // Radial falloff as stand-in for IES profile (no .ies file load)
                let strength = inp(1, SocketVal::Scalar(if node.value.abs() < 1e-6 { 1.0 } else { node.value })).as_scalar();
                if live {
                    let (col, fac) = proc_sample(node.kind, node.mode, strength, &node.props, node.color, 0.37, 0.61);
                    cache.insert((id, 0), SocketVal::Scalar(fac * strength));
                    cache.insert((id, 1), SocketVal::Color([col[0]*strength, col[1]*strength, col[2]*strength, 1.0]));
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
                        rgba[i] = b; rgba[i+1] = b; rgba[i+2] = b; rgba[i+3] = 255;
                    }
                }
                let tex = TextureData { name: format!("IES_{id}"), width: size, height: size, rgba8: rgba };
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
                cache.insert((id, 0), SocketVal::Color(mix4(a, b, fac)));
            }
            ShaderNodeKind::Blackbody => {
                let t = inp(0, SocketVal::Scalar(if node.value > 100.0 { node.value } else { 6500.0 })).as_scalar();
                // Rough Planck approximation to RGB
                let x = (t / 1000.0).clamp(1.0, 40.0);
                let r = if x <= 6.6 { 1.0 } else { (1.292_936 * (x - 6.0).powf(-0.133)).clamp(0.0, 1.0) };
                let g = if x <= 6.6 {
                    (0.390_081 * x.ln() - 0.631).clamp(0.0, 1.0)
                } else {
                    (1.196_040 * (x - 6.0).powf(-0.075)).clamp(0.0, 1.0)
                };
                let b = if x >= 6.6 { 1.0 } else if x <= 1.9 { 0.0 } else {
                    (0.543_206 * (x - 1.0).ln() - 0.370).clamp(0.0, 1.0)
                };
                cache.insert((id, 0), SocketVal::Color([r, g, b, 1.0]));
            }
            ShaderNodeKind::Wavelength => {
                let nm = inp(0, SocketVal::Scalar(if node.value > 10.0 { node.value } else { 550.0 })).as_scalar();
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
                let contrast = inp(2, SocketVal::Scalar(if node.props[1].abs() < 1e-6 { 0.0 } else { node.props[1] })).as_scalar();
                let f = |x: f32| ((x - 0.5) * (1.0 + contrast) + 0.5 + bright).clamp(0.0, 1.0);
                cache.insert((id, 0), SocketVal::Color([f(c[0]), f(c[1]), f(c[2]), c[3]]));
            }
            ShaderNodeKind::Gamma => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                let g = inp(1, SocketVal::Scalar(if node.value.abs() < 1e-6 { 1.0 } else { node.value })).as_scalar().max(0.01);
                cache.insert((id, 0), SocketVal::Color([c[0].powf(g), c[1].powf(g), c[2].powf(g), c[3]]));
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
                let expr = if node.name_prop.is_empty() { "a" } else { node.name_prop.as_str() };
                let r = eval_simple_expr(expr, a, b);
                cache.insert((id, 0), SocketVal::Color([r, r, r, 1.0]));
            }
            ShaderNodeKind::MapRange => {
                let v = inp(0, SocketVal::Scalar(node.value)).as_scalar();
                let fmin = inp(1, SocketVal::Scalar(0.0)).as_scalar();
                let fmax = inp(2, SocketVal::Scalar(1.0)).as_scalar();
                let tmin = inp(3, SocketVal::Scalar(0.0)).as_scalar();
                let tmax = inp(4, SocketVal::Scalar(1.0)).as_scalar();
                let t = if (fmax - fmin).abs() < 1e-8 { 0.0 } else { (v - fmin) / (fmax - fmin) };
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
                        let l = (a[0]*a[0]+a[1]*a[1]+a[2]*a[2]).sqrt().max(1e-8);
                        [a[0]/l, a[1]/l, a[2]/l]
                    }
                    5 => {
                        // cross
                        [
                            a[1]*b[2] - a[2]*b[1],
                            a[2]*b[0] - a[0]*b[2],
                            a[0]*b[1] - a[1]*b[0],
                        ]
                    }
                    6 => {
                        let d = a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
                        [d, d, d]
                    }
                    7 => [a[0].abs(), a[1].abs(), a[2].abs()],
                    8 => [-a[0], -a[1], -a[2]],
                    9 => [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])],
                    10 => [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])],
                    11 => {
                        let l = (a[0]*a[0]+a[1]*a[1]+a[2]*a[2]).sqrt();
                        [l, l, l]
                    }
                    _ => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
                };
                cache.insert((id, 0), SocketVal::Vector(r));
            }
            ShaderNodeKind::Bump => {
                let strength = inp(0, SocketVal::Scalar(if node.value.abs() < 1e-6 { 1.0 } else { node.value })).as_scalar();
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
                let ior = inp(0, SocketVal::Scalar(if node.value.abs() < 1e-6 { 1.45 } else { node.value })).as_scalar();
                let f0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
                let f = f0 + (1.0 - f0) * 0.5;
                cache.insert((id, 0), SocketVal::Scalar(f.clamp(0.0, 1.0)));
                cache.insert((id, 1), SocketVal::Scalar((1.0 - f).clamp(0.0, 1.0)));
            }
            ShaderNodeKind::Emission | ShaderNodeKind::Background => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                let strength = inp(1, SocketVal::Scalar(if node.value.abs() < 1e-6 { 1.0 } else { node.value })).as_scalar();
                cache.insert((id, 0), SocketVal::Color([c[0]*strength, c[1]*strength, c[2]*strength, c[3]]));
            }
            ShaderNodeKind::Diffuse | ShaderNodeKind::Glossy | ShaderNodeKind::Transparent
            | ShaderNodeKind::Translucent | ShaderNodeKind::Sheen | ShaderNodeKind::Glass
            | ShaderNodeKind::Refraction | ShaderNodeKind::Toon | ShaderNodeKind::Subsurface
            | ShaderNodeKind::Metallic | ShaderNodeKind::Specular | ShaderNodeKind::Hair
            | ShaderNodeKind::Holdout => {
                let c = inp(0, SocketVal::Color(node.color)).as_color();
                cache.insert((id, 0), SocketVal::Color(c));
            }
            ShaderNodeKind::AddShader | ShaderNodeKind::MixShader => {
                let a = inp(if node.kind == ShaderNodeKind::MixShader { 1 } else { 0 }, SocketVal::Color([0.0,0.0,0.0,1.0])).as_color();
                let b = inp(if node.kind == ShaderNodeKind::MixShader { 2 } else { 1 }, SocketVal::Color([0.0,0.0,0.0,1.0])).as_color();
                let fac = if node.kind == ShaderNodeKind::MixShader {
                    inp(0, SocketVal::Scalar(0.5)).as_scalar()
                } else { 0.5 };
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
                let roughness = inp(2, SocketVal::Scalar(if node.props[1].abs() < 1e-6 { 0.5 } else { node.props[1] })).as_scalar();
                let ior = inp(3, SocketVal::Scalar(if node.props[2].abs() < 1e-6 { 1.45 } else { node.props[2] })).as_scalar();
                let alpha = inp(4, SocketVal::Scalar(if node.props[3].abs() < 1e-6 { 1.0 } else { node.props[3] })).as_scalar();
                let emission = inp(6, SocketVal::Color([0.0, 0.0, 0.0, 1.0])).as_color();
                let em_str = inp(7, SocketVal::Scalar(0.0)).as_scalar();
                let ao = inp(13, SocketVal::Scalar(1.0)).as_scalar();
                result.base_color = [base[0], base[1], base[2], alpha];
                result.metallic = metallic.clamp(0.0, 1.0);
                result.roughness = roughness.clamp(0.045, 1.0);
                result.ior = ior.max(1.0);
                result.alpha = alpha.clamp(0.0, 1.0);
                result.emissive = [emission[0] * em_str, emission[1] * em_str, emission[2] * em_str];
                result.emission_strength = em_str;
                result.occlusion_strength = ao.clamp(0.0, 1.0);
                // Texture wiring from linked image/noise nodes
                for link in &graph.links {
                    if link.to != id { continue; }
                    if let Some(src) = node_map.get(&link.from) {
                        match src.kind {
                            ShaderNodeKind::ImageTexture | ShaderNodeKind::Noise | ShaderNodeKind::Voronoi
                            | ShaderNodeKind::Wave | ShaderNodeKind::Checker | ShaderNodeKind::Brick
                            | ShaderNodeKind::Magic | ShaderNodeKind::Gradient | ShaderNodeKind::WhiteNoise => {
                                if let Some(&ti) = proc_tex_index.get(&src.id) {
                                    match link.to_socket {
                                        0 => result.base_color_tex = Some(ti),
                                        5 | 13 => result.normal_tex = Some(ti), // Normal / AO
                                        6 => result.emissive_tex = Some(ti),
                                        _ => {}
                                    }
                                }
                            }
                            ShaderNodeKind::NormalMap => {
                                if let Some(ti) = src.image_index { result.normal_tex = Some(ti); }
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
            ShaderNodeKind::VolumeAbsorption | ShaderNodeKind::VolumeScatter
            | ShaderNodeKind::VolumeCoefficients | ShaderNodeKind::PrincipledVolume => {
                let dens = inp(1, SocketVal::Scalar(node.value)).as_scalar();
                result.volume_density = dens;
                cache.insert((id, 0), SocketVal::Color(inp(0, SocketVal::Color(node.color)).as_color()));
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
        if matches!(n.kind, ShaderNodeKind::Output | ShaderNodeKind::WorldOutput | ShaderNodeKind::LightOutput) {
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

fn sample_graph_at(
    graph: &ShaderGraph,
    node_map: &HashMap<u64, &ShaderNode>,
    textures: &[TextureData],
    base: &GraphEvalResult,
    u: f32,
    v: f32,
) -> [f32; 4] {
    let order = topo_order(graph);
    let mut cache: HashMap<(u64, u8), SocketVal> = HashMap::new();
    for id in order {
        let Some(node) = node_map.get(&id) else { continue };
        match node.kind {
            ShaderNodeKind::TexCoord | ShaderNodeKind::UvMap => {
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
            ShaderNodeKind::Noise | ShaderNodeKind::Wave | ShaderNodeKind::Magic => {
                let uv = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Vector([u, v, 0.0]))
                    .as_vector();
                let scale = if node.value.abs() < 1e-4 { 5.0 } else { node.value };
                let n = fbm(uv[0] * scale, uv[1] * scale, 5);
                cache.insert((id, 0), SocketVal::Color([n, n, n, 1.0]));
            }
            ShaderNodeKind::Voronoi | ShaderNodeKind::WhiteNoise => {
                let uv = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Vector([u, v, 0.0]))
                    .as_vector();
                let scale = if node.value.abs() < 1e-4 { 5.0 } else { node.value };
                let n = voronoi(uv[0] * scale, uv[1] * scale);
                cache.insert((id, 0), SocketVal::Color([n, n, n, 1.0]));
            }
            ShaderNodeKind::Checker | ShaderNodeKind::Brick => {
                let scale = if node.value.abs() < 1e-4 { 4.0 } else { node.value };
                let cx = (u * scale).floor() as i32;
                let cy = (v * scale).floor() as i32;
                let n = if (cx + cy) % 2 == 0 { 1.0 } else { 0.0 };
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
                let col = color_ramp_from_node(node, fac);
                cache.insert((id, 0), SocketVal::Color(col));
                cache.insert((id, 1), SocketVal::Scalar(col[3]));
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
            ShaderNodeKind::Principled | ShaderNodeKind::Diffuse | ShaderNodeKind::Glossy
            | ShaderNodeKind::Emission | ShaderNodeKind::Transparent => {
                let c = find_input(graph, id, 0, &cache)
                    .unwrap_or(SocketVal::Color(base.base_color))
                    .as_color();
                cache.insert((id, 0), SocketVal::Color(c));
            }
            ShaderNodeKind::Output => {
                if let Some(s) = find_input(graph, id, 0, &cache) {
                    cache.insert((id, 0), s);
                }
            }
            ShaderNodeKind::Fresnel | ShaderNodeKind::LayerWeight => {
                // Approximate facing from UV distance to center (bake-friendly).
                let dx = u - 0.5;
                let dy = v - 0.5;
                let facing = (1.0 - (dx * dx + dy * dy).sqrt() * 2.0).clamp(0.0, 1.0);
                let ior = if node.value.abs() < 1e-6 { 1.45 } else { node.value };
                let f0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
                let f = f0 + (1.0 - f0) * (1.0 - facing).powi(5);
                cache.insert((id, 0), SocketVal::Scalar(f.clamp(0.0, 1.0)));
                cache.insert((id, 1), SocketVal::Scalar(facing));
            }
            ShaderNodeKind::Geometry => {
                cache.insert((id, 0), SocketVal::Vector([u, v, 0.0])); // Position ~ UV
                // Normal from finite differences of a flat plane
                cache.insert((id, 1), SocketVal::Vector([0.0, 0.0, 1.0]));
                cache.insert((id, 2), SocketVal::Vector([1.0, 0.0, 0.0]));
            }
            ShaderNodeKind::AmbientOcclusion => {
                // Cheap curvature-ish AO from noise
                let n = fbm(u * 4.0, v * 4.0, 3);
                let ao = (0.55 + 0.45 * n).clamp(0.0, 1.0);
                cache.insert((id, 0), SocketVal::Color([ao, ao, ao, 1.0]));
            }
            ShaderNodeKind::Wireframe => {
                let size = if node.value.abs() < 1e-6 { 0.02 } else { node.value.abs() };
                let fu = u.fract();
                let fv = v.fract();
                let edge = (fu.min(1.0 - fu).min(fv.min(1.0 - fv)) < size) as i32 as f32;
                cache.insert((id, 0), SocketVal::Scalar(edge));
            }
            ShaderNodeKind::Math => {
                let a = find_input(graph, id, 0, &cache).unwrap_or(SocketVal::Scalar(node.value)).as_scalar();
                let b = find_input(graph, id, 1, &cache).unwrap_or(SocketVal::Scalar(node.props[0])).as_scalar();
                cache.insert((id, 0), SocketVal::Scalar(math_op(node.mode, a, b)));
            }
            ShaderNodeKind::Gamma => {
                let c = find_input(graph, id, 0, &cache).unwrap_or(SocketVal::Color(node.color)).as_color();
                let g = find_input(graph, id, 1, &cache).unwrap_or(SocketVal::Scalar(1.0)).as_scalar().max(0.01);
                cache.insert((id, 0), SocketVal::Color([c[0].powf(g), c[1].powf(g), c[2].powf(g), c[3]]));
            }
            ShaderNodeKind::InvertColor => {
                let fac = find_input(graph, id, 0, &cache).unwrap_or(SocketVal::Scalar(1.0)).as_scalar();
                let c = find_input(graph, id, 1, &cache).unwrap_or(SocketVal::Color(node.color)).as_color();
                let inv = [1.0 - c[0], 1.0 - c[1], 1.0 - c[2], c[3]];
                cache.insert((id, 0), SocketVal::Color(mix4(c, inv, fac)));
            }
            _ => {}
        }
    }
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







/// Fast single-UV sample for live viewport (no texture allocation).
fn proc_sample(kind: ShaderNodeKind, mode: u8, value: f32, props: &[f32; 16], color: [f32; 4], u: f32, v: f32) -> ([f32; 4], f32) {
    let scale = if value.abs() < 1e-4 { 5.0 } else { value.abs() };
    match kind {
        ShaderNodeKind::Noise | ShaderNodeKind::Magic | ShaderNodeKind::Gabor => {
            let detail = if props[0] > 0.0 { props[0] } else { 4.0 };
            let n = fbm(u * scale, v * scale, detail.clamp(1.0, 8.0) as i32);
            ([n, n, n, 1.0], n)
        }
        ShaderNodeKind::Voronoi | ShaderNodeKind::WhiteNoise => {
            let (d, col) = voronoi_ex(u * scale, v * scale, if props[0] > 0.0 { props[0] } else { 1.0 }, mode);
            ([col[0], col[1], col[2], 1.0], d)
        }
        ShaderNodeKind::Checker => {
            let cx = (u * scale).floor() as i32;
            let cy = (v * scale).floor() as i32;
            let fac = if (cx + cy) % 2 == 0 { 1.0 } else { 0.0 };
            let c = if fac > 0.5 { [1.0, 1.0, 1.0, 1.0] } else { [0.0, 0.0, 0.0, 1.0] };
            let _ = color;
            (c, fac)
        }
        ShaderNodeKind::Brick => {
            let row_h = 0.25;
            let brick_w = 0.5;
            let mortar = 0.08;
            let row = (v * scale / row_h).floor();
            let offset = if (row as i32) % 2 == 0 { 0.0 } else { brick_w * 0.5 };
            let bx = (u * scale + offset) / brick_w;
            let fx = bx.fract();
            let fy = (v * scale / row_h).fract();
            let in_mortar = fx < mortar || fx > 1.0 - mortar || fy < mortar || fy > 1.0 - mortar;
            if in_mortar {
                ([0.35, 0.35, 0.35, 1.0], 0.0)
            } else {
                ([0.6, 0.25, 0.15, 1.0], 1.0)
            }
        }
        ShaderNodeKind::Gradient => {
            let t = match mode {
                1 => u * u,
                2 => {
                    let x = u.clamp(0.0, 1.0);
                    x * x * (3.0 - 2.0 * x)
                }
                3 => ((u + v) * 0.5).clamp(0.0, 1.0),
                4 => {
                    let dx = u - 0.5;
                    let dy = v - 0.5;
                    (1.0 - (dx * dx + dy * dy).sqrt() * 2.0).clamp(0.0, 1.0)
                }
                5 => {
                    let a = (v - 0.5).atan2(u - 0.5);
                    (a / (2.0 * std::f32::consts::PI) + 0.5).rem_euclid(1.0)
                }
                _ => u.clamp(0.0, 1.0),
            };
            ([t, t, t, 1.0], t)
        }
        ShaderNodeKind::Wave => {
            let bands = mode % 2 == 0;
            let profile = (mode / 2) % 3;
            let p = if bands {
                u * scale
            } else {
                let dx = u - 0.5;
                let dy = v - 0.5;
                (dx * dx + dy * dy).sqrt() * scale
            };
            let t = match profile {
                1 => p.fract(),
                2 => {
                    let f = p.fract();
                    if f < 0.5 { f * 2.0 } else { 2.0 - f * 2.0 }
                }
                _ => (p * std::f32::consts::TAU).sin() * 0.5 + 0.5,
            };
            ([t, t, t, 1.0], t)
        }
        ShaderNodeKind::Sky => ([0.45 + 0.2 * (1.0 - v), 0.55 + 0.15 * (1.0 - v), 0.85 - 0.1 * v, 1.0], 1.0),
        ShaderNodeKind::Ies => {
            let dx = u - 0.5;
            let dy = v - 0.5;
            let t = ((1.0 - (dx * dx + dy * dy).sqrt() * 2.0).max(0.0).powf(2.0)).clamp(0.0, 1.0);
            ([t, t, t, 1.0], t)
        }
        _ => ([0.5, 0.5, 0.5, 1.0], 0.5),
    }
}

fn voronoi_ex(x: f32, y: f32, randomness: f32, mode: u8) -> (f32, [f32; 3]) {
    let ix = x.floor();
    let iy = y.floor();
    let mut best = 1.0e9f32;
    let mut second = 1.0e9f32;
    let mut best_col = [0.5f32; 3];
    for j in -1..=1 {
        for i in -1..=1 {
            let cx = ix + i as f32;
            let cy = iy + j as f32;
            let h = hash2(cx, cy);
            let ox = cx + (hash2(cx + 1.7, cy + 3.1) - 0.5) * randomness + 0.5;
            let oy = cy + (hash2(cx + 5.3, cy + 2.9) - 0.5) * randomness + 0.5;
            let d = ((x - ox).powi(2) + (y - oy).powi(2)).sqrt();
            if d < best {
                second = best;
                best = d;
                best_col = [h, hash2(cx + 9.1, cy), hash2(cx, cy + 4.4)];
            } else if d < second {
                second = d;
            }
        }
    }
    let dist = match mode {
        1 => second.clamp(0.0, 1.0), // F2
        2 => (second - best).clamp(0.0, 1.0), // Distance to edge approx
        _ => best.clamp(0.0, 1.0), // F1
    };
    (dist, best_col)
}

fn math_op(mode: u8, a: f32, b: f32) -> f32 {
    match mode {
        1 => a - b,
        2 => a * b,
        3 => if b.abs() > 1e-8 { a / b } else { 0.0 },
        4 => a.powf(b),
        5 => a.log(b.max(1e-8)),
        6 => a.sqrt().max(0.0),
        7 => a.abs(),
        8 => a.min(b),
        9 => a.max(b),
        10 => a.sin(),
        11 => a.cos(),
        12 => a.tan(),
        13 => a.floor(),
        14 => a.ceil(),
        15 => a.round(),
        16 => a.fract(),
        17 => a.clamp(0.0, 1.0),
        18 => -a,
        19 => a % b,
        20 => if a < b { 1.0 } else { 0.0 },
        21 => if a > b { 1.0 } else { 0.0 },
        22 => if (a - b).abs() < 1e-5 { 1.0 } else { 0.0 },
        23 => a.asin(),
        24 => a.acos(),
        25 => a.atan(),
        26 => a.atan2(b),
        27 => a.sinh(),
        28 => a.cosh(),
        29 => a.tanh(),
        30 => a * std::f32::consts::PI / 180.0, // to radians
        31 => a * 180.0 / std::f32::consts::PI, // to degrees
        32 => a.exp(),
        33 => if a > 1e-8 { 1.0 / a.sqrt() } else { 0.0 },
        34 => {
            // smooth min
            let k = b.max(1e-5);
            let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
            a * h + b * (1.0 - h) - k * h * (1.0 - h)
        }
        35 => {
            let k = b.max(1e-5);
            let h = (0.5 + 0.5 * (a - b) / k).clamp(0.0, 1.0);
            a * h + b * (1.0 - h) + k * h * (1.0 - h)
        }
        36 => {
            // wrap
            let r = b;
            if r.abs() < 1e-8 { a } else { a - r * (a / r).floor() }
        }
        37 => {
            // snap
            if b.abs() < 1e-8 { a } else { (a / b).floor() * b }
        }
        38 => {
            // ping-pong
            let r = b.abs().max(1e-8);
            let t = a.rem_euclid(r * 2.0);
            if t < r { t } else { r * 2.0 - t }
        }
        39 => a.signum(),
        40 => a * a,
        _ => a + b, // Add
    }
}

/// Tiny expression evaluator: supports a, b, + - * / ( ), numbers.
fn eval_simple_expr(expr: &str, a: f32, b: f32) -> f32 {
    let s = expr.trim().to_lowercase().replace(" ", "");
    if s.is_empty() { return a; }
    // Replace identifiers
    let mut t = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_alphabetic() {
            let mut j = i;
            while j < chars.len() && chars[j].is_ascii_alphanumeric() { j += 1; }
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
    struct P<'a> { s: &'a [u8], i: usize }
    impl<'a> P<'a> {
        fn peek(&self) -> Option<u8> { self.s.get(self.i).copied() }
        fn get(&mut self) -> Option<u8> { let c = self.peek()?; self.i += 1; Some(c) }
        fn skip(&mut self) { while matches!(self.peek(), Some(b' ')) { self.i += 1; } }
        fn number(&mut self) -> Option<f32> {
            self.skip();
            let start = self.i;
            if self.peek() == Some(b'-') && start == self.i { self.i += 1; }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == b'.') { self.i += 1; }
            if self.i == start { return None; }
            std::str::from_utf8(&self.s[start..self.i]).ok()?.parse().ok()
        }
        fn factor(&mut self) -> Option<f32> {
            self.skip();
            if self.peek() == Some(b'(') {
                self.get();
                let v = self.expr()?;
                self.skip();
                if self.get() != Some(b')') { return None; }
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
                    Some(b'*') => { self.get(); v *= self.factor()?; }
                    Some(b'/') => { self.get(); let d = self.factor()?; if d.abs() > 1e-12 { v /= d; } }
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
                    Some(b'+') => { self.get(); v += self.term()?; }
                    Some(b'-') => { self.get(); v -= self.term()?; }
                    _ => break,
                }
            }
            Some(v)
        }
    }
    let mut p = P { s: s.as_bytes(), i: 0 };
    let v = p.expr()?;
    p.skip();
    if p.i != p.s.len() { return None; }
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
