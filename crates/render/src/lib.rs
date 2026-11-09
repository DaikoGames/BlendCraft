//! wgpu renderer. Targets: Vulkan / Metal / DX12 / GL (native) and WebGPU / WebGL2 (wasm).
//!
//! Step 2 features: glTF PBR materials + textures (with mips), image-based lighting from an
//! equirect HDR (SH diffuse + roughness-mip specular), procedural sky, view modes, wireframe.
mod settings;
mod texture;

pub use settings::{Highlight, ShadingMode, ViewSettings, WireMode};

use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use forge_core::{AlphaMode, EnvironmentMap, LightType, Material, MeshData, ObjectKind, OrbitCamera, Scene, TextureData, Vertex};
use glam::{Mat3, Mat4};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const ENV_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    env: [f32; 4],
    opts: [f32; 4],
    sh: [[f32; 4]; 9],
    light_info: [f32; 4],
    lights: [[[f32; 4]; 4]; 8],
    light_view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialUniform {
    base_color: [f32; 4],
    emissive: [f32; 4],
    params: [f32; 4],
    flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LineVertex {
    pos: [f32; 3],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ModelUniform {
    matrix: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    tint: [f32; 4],
}

/// GPU buffers for one mesh; shared between objects that share an `Arc<MeshData>`.
struct MeshBuffers {
    vb: wgpu::Buffer,
    ib: wgpu::Buffer,
    index_count: u32,
    wire_ib: wgpu::Buffer,
    wire_count: u32,
}

/// One draw call range of an object (an object has several when its mesh has material groups).
struct Draw {
    first: u32,
    count: u32,
    material: usize,
    blend: bool,
}

struct GpuObject {
    buffers: Arc<MeshBuffers>,
    model_buf: wgpu::Buffer,
    model_bg: wgpu::BindGroup,
    draws: Vec<Draw>,
    highlight: Highlight,
}

struct GpuMaterial {
    bind_group: wgpu::BindGroup,
    blend: bool,
}

const MESH_ATTRS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
const POS_ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];
const LINE_ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

type TexCache = HashMap<(usize, bool), wgpu::TextureView>;

const SHADOW_SIZE: u32 = 4096;

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,

    global_buf: wgpu::Buffer,
    global_bgl: wgpu::BindGroupLayout,
    global_bg: wgpu::BindGroup,
    material_bgl: wgpu::BindGroupLayout,
    model_bgl: wgpu::BindGroupLayout,
    env_sampler: wgpu::Sampler,
    mat_sampler: wgpu::Sampler,
    env_max_lod: f32,
    sh: [[f32; 4]; 9],
    light_count: u32,
    lights: [[[f32; 4]; 4]; 8],
    /// Index of the light used for shadows (-1 = none)
    shadow_light: i32,
    light_view_proj: Mat4,

    #[allow(dead_code)]
    shadow_tex: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    shadow_sampler: wgpu::Sampler,
    shadow_pipeline: wgpu::RenderPipeline,
    /// Globals bind group WITHOUT the shadow map (depth pass only).
    shadow_global_bg: wgpu::BindGroup,

    white: wgpu::TextureView,
    flat_normal: wgpu::TextureView,

    sky_pipeline: wgpu::RenderPipeline,
    grid_pipeline: wgpu::RenderPipeline,
    mesh_opaque: wgpu::RenderPipeline,
    mesh_blend: wgpu::RenderPipeline,
    wire_pipeline: wgpu::RenderPipeline,

    grid_vb: wgpu::Buffer,
    grid_count: u32,
    materials: Vec<GpuMaterial>,
    objects: Vec<GpuObject>,
    mesh_cache: HashMap<usize, (Arc<MeshData>, Arc<MeshBuffers>)>,
    tex_cache: TexCache,
    tex_owner: Option<Arc<Vec<TextureData>>>,
    /// Reused offscreen targets for capture (avoid alloc per frame).
    capture_cache: Option<CaptureCache>,
}

struct CaptureCache {
    width: u32,
    height: u32,
    color_tex: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    staging: wgpu::Buffer,
    padded_bpr: u32,
}

impl Renderer {
    /// `target` is anything winit's `Arc<Window>` converts into (or a canvas on wasm).
    pub async fn new(target: impl Into<wgpu::SurfaceTarget<'static>>, width: u32, height: u32) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(target)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow!("no suitable GPU adapter found"))?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("forge device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        let depth_view = create_depth(&device, config.width, config.height);

        // ---- bind group layouts ----
        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let uniform_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let global_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("global bgl"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
                tex_entry(1),
                sampler_entry(2),
                // shadow map (depth) + comparison sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material bgl"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                tex_entry(1),
                tex_entry(2),
                tex_entry(3),
                tex_entry(4),
                tex_entry(5),
                sampler_entry(6),
            ],
        });
        let model_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model bgl"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT)],
        });
        let layout_wire = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout wire"),
            bind_group_layouts: &[&global_bgl, &model_bgl],
            push_constant_ranges: &[],
        });
        let layout_basic = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout basic"),
            bind_group_layouts: &[&global_bgl],
            push_constant_ranges: &[],
        });
        let layout_mat = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout material"),
            bind_group_layouts: &[&global_bgl, &material_bgl, &model_bgl],
            push_constant_ranges: &[],
        });

        // ---- samplers ----
        let env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("env sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mat_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });

        // ---- fallback textures ----
        let one = |rgba: [u8; 4], label: &str| {
            texture::upload_mips(
                &device,
                &queue,
                label,
                &[texture::MipLevel { w: 1, h: 1, data: rgba.to_vec() }],
                wgpu::TextureFormat::Rgba8Unorm,
            )
        };
        let white = one([255, 255, 255, 255], "white");
        let flat_normal = one([128, 128, 255, 255], "flat normal");

        // ---- globals + environment ----
        let global_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (env_view, sh, env_max_lod) =
            build_environment(&device, &queue, &EnvironmentMap::procedural_sky(512, 256));

        // ---- shadow map (single depth map) ----
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow map"),
            size: wgpu::Extent3d { width: SHADOW_SIZE, height: SHADOW_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let global_bg = make_global_bg(
            &device, &global_bgl, &global_buf, &env_view, &env_sampler, &shadow_view, &shadow_sampler,
        );

        // ---- pipelines ----
        let src = |body: &str, common: &str| format!("{common}\n{body}");
        let common = include_str!("common.wgsl");
        let vertex_buf = || wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &MESH_ATTRS,
        };
        let sky_pipeline = make_pipeline(
            &device, &layout_basic, format, &src(include_str!("sky.wgsl"), common), &[],
            wgpu::PrimitiveTopology::TriangleList, false, wgpu::CompareFunction::Always, wgpu::BlendState::REPLACE,
        );
        let grid_pipeline = make_pipeline(
            &device, &layout_basic, format, &src(include_str!("grid.wgsl"), common),
            &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LineVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &LINE_ATTRS,
            }],
            wgpu::PrimitiveTopology::LineList, true, wgpu::CompareFunction::Less, wgpu::BlendState::REPLACE,
        );
        let mesh_src = src(include_str!("mesh.wgsl"), common);
        let mesh_opaque = make_pipeline(
            &device, &layout_mat, format, &mesh_src, &[vertex_buf()],
            wgpu::PrimitiveTopology::TriangleList, true, wgpu::CompareFunction::Less, wgpu::BlendState::REPLACE,
        );
        let mesh_blend = make_pipeline(
            &device, &layout_mat, format, &mesh_src, &[vertex_buf()],
            wgpu::PrimitiveTopology::TriangleList, false, wgpu::CompareFunction::Less, wgpu::BlendState::ALPHA_BLENDING,
        );
        let wire_pipeline = make_pipeline(
            &device, &layout_wire, format, &src(include_str!("wire.wgsl"), common),
            &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &POS_ATTRS,
            }],
            wgpu::PrimitiveTopology::LineList, false, wgpu::CompareFunction::LessEqual, wgpu::BlendState::REPLACE,
        );

        // Depth-only shadow pipeline: minimal group 0 layout so we never bind the
        // shadow map texture while writing to it (wgpu exclusive usage rule).
        let shadow_global_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow global bgl"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX)],
        });
        let shadow_global_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow global bg"),
            layout: &shadow_global_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: global_buf.as_entire_binding(),
            }],
        });
        // Minimal shader: only needs light_view_proj from Globals + model matrix.
        // Layout must match the CPU-side Globals struct up through light_view_proj.
        let shadow_src = r#"
struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    env: vec4<f32>,
    opts: vec4<f32>,
    sh: array<vec4<f32>, 9>,
    light_info: vec4<f32>,
    lights: array<array<vec4<f32>, 4>, 8>,
    light_view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> g: Globals;
struct Model {
    matrix: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    tint: vec4<f32>,
};
@group(1) @binding(0) var<uniform> model: Model;
@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    let wp = model.matrix * vec4<f32>(pos, 1.0);
    return g.light_view_proj * wp;
}
"#;
        let shadow_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shadow_src)),
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow pl"),
                bind_group_layouts: &[&shadow_global_bgl, &model_bgl],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module: &shadow_module,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &POS_ATTRS,
                }],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // No culling: single-sided planes and thin meshes must still cast shadows
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                // Constant + slope bias reduces shadow acne on grazing surfaces
                bias: wgpu::DepthBiasState {
                    constant: 0,
                    slope_scale: 1.25,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let grid = build_grid(20, 1.0);
        let grid_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid"),
            contents: bytemuck::cast_slice(&grid),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Ok(Self {
            surface, device, queue, config, depth_view,
            global_buf, global_bgl, global_bg, material_bgl, model_bgl, env_sampler, mat_sampler, env_max_lod, sh,
            light_count: 0, lights: [[[0.0; 4]; 4]; 8],
            shadow_light: -1,
            light_view_proj: Mat4::IDENTITY,
            shadow_tex, shadow_view, shadow_sampler, shadow_pipeline, shadow_global_bg,
            white, flat_normal,
            sky_pipeline, grid_pipeline, mesh_opaque, mesh_blend, wire_pipeline,
            grid_vb, grid_count: grid.len() as u32, materials: Vec::new(), objects: Vec::new(),
            mesh_cache: HashMap::new(), tex_cache: HashMap::new(), tex_owner: None,
            capture_cache: None,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth(&self.device, width, height);
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Swap the lighting environment (equirect, linear HDR).
    pub fn set_environment(&mut self, env: &EnvironmentMap) {
        let (view, sh, max_lod) = build_environment(&self.device, &self.queue, env);
        self.global_bg = make_global_bg(
            &self.device, &self.global_bgl, &self.global_buf, &view, &self.env_sampler,
            &self.shadow_view, &self.shadow_sampler,
        );
        self.sh = sh;
        self.env_max_lod = max_lod;
    }

    /// Sync the GPU state with the scene. Textures and mesh buffers are cached (by `Arc` identity),
    /// so calling this after every small edit (e.g. dragging a transform slider) is cheap.
    /// `highlight[i]` says how object `i` is outlined (selection state).
    pub fn set_scene(&mut self, scene: &Scene, highlight: &[Highlight]) {
        // textures
        let same = self.tex_owner.as_ref().map_or(false, |a| Arc::ptr_eq(a, &scene.textures));
        if !same {
            self.tex_cache.clear();
            self.tex_owner = Some(scene.textures.clone());
        }
        let mut cache = std::mem::take(&mut self.tex_cache);
        let mut mats = vec![self.make_material(&Material::default(), scene, &mut cache)];
        for m in &scene.materials {
            mats.push(self.make_material(m, scene, &mut cache));
        }
        self.tex_cache = cache;

        let worlds = scene.world_matrices();

        // meshes + objects
        let mut old = std::mem::take(&mut self.mesh_cache);
        let mut fresh: HashMap<usize, (Arc<MeshData>, Arc<MeshBuffers>)> = HashMap::new();
        let mut objects = Vec::new();
        for (i, o) in scene.objects.iter().enumerate() {
            if !scene.is_visible(i) || o.mesh.indices.is_empty() {
                continue;
            }
            let key = Arc::as_ptr(&o.mesh) as usize;
            let buffers = if let Some((_, b)) = fresh.get(&key) {
                b.clone()
            } else {
                let b = match old.remove(&key) {
                    Some((_, b)) => b,
                    None => Arc::new(self.upload_mesh(&o.mesh)),
                };
                fresh.insert(key, (o.mesh.clone(), b.clone()));
                b
            };
            let slot = |m: Option<usize>| m.map(|m| m + 1).filter(|&m| m < mats.len()).unwrap_or(0);
            let mut draws = Vec::new();
            if o.mesh.groups.is_empty() {
                let mi = slot(o.material);
                draws.push(Draw { first: 0, count: buffers.index_count, material: mi, blend: mats[mi].blend });
            } else {
                for g in &o.mesh.groups {
                    let mi = slot(g.material);
                    draws.push(Draw { first: g.first_index, count: g.index_count, material: mi, blend: mats[mi].blend });
                }
            }
            let hl = highlight.get(i).copied().unwrap_or(Highlight::None);
            let mu = model_uniform(&worlds[i], hl);
            let model_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("model uniform"),
                contents: bytemuck::bytes_of(&mu),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let model_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("model bg"),
                layout: &self.model_bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: model_buf.as_entire_binding() }],
            });
            objects.push(GpuObject { buffers, model_buf, model_bg, draws, highlight: hl });
        }
        self.mesh_cache = fresh;
        self.materials = mats;
        self.objects = objects;

        // punctual lights (max 8)
        let mut lights = [[[0.0f32; 4]; 4]; 8];
        let mut n = 0usize;
        // Track candidates for the shadow-casting light (any type can cast now).
        let mut shadow_idx = -1i32;
        let mut shadow_kind = LightType::Sun;
        let mut shadow_dir = glam::Vec3::NEG_Y;
        let mut shadow_pos = glam::Vec3::ZERO;
        let mut shadow_spot_angle = 45.0f32;
        let mut best_priority = -1i32; // higher = preferred (Sun=3, Spot=2, Point=1)
        for (i, o) in scene.objects.iter().enumerate() {
            if !scene.is_visible(i) {
                continue;
            }
            if let ObjectKind::Light(l) = &o.kind {
                if n >= 8 {
                    break;
                }
                let w = worlds[i];
                let pos = w.w_axis.truncate();
                let dir = w.transform_vector3(glam::Vec3::NEG_Z).normalize_or_zero();
                let ty = match l.kind {
                    LightType::Point | LightType::Area => 0.0,
                    LightType::Sun => 1.0,
                    LightType::Spot => 2.0,
                };
                let outer = (l.spot_angle_deg * 0.5).to_radians().clamp(0.01, 1.55);
                let inner = outer * (1.0 - l.spot_blend.clamp(0.0, 0.99));
                lights[n] = [
                    [pos.x, pos.y, pos.z, ty],
                    [l.color[0] * l.intensity, l.color[1] * l.intensity, l.color[2] * l.intensity, l.range],
                    [dir.x, dir.y, dir.z, 0.0],
                    [outer.cos(), inner.cos(), 0.0, 0.0],
                ];
                // Active light always wins so the light you are editing is the one that casts.
                // Local lights (Spot/Point) beat Sun so a lamp above the scene actually shadows.
                let is_active = matches!(highlight.get(i).copied(), Some(Highlight::Active));
                let prio = if is_active {
                    100
                } else {
                    match l.kind {
                        LightType::Spot => 3,
                        LightType::Point | LightType::Area => 2,
                        LightType::Sun => 1,
                    }
                };
                if prio > best_priority {
                    best_priority = prio;
                    shadow_idx = n as i32;
                    shadow_kind = l.kind;
                    shadow_dir = dir;
                    shadow_pos = pos;
                    shadow_spot_angle = l.spot_angle_deg;
                }
                n += 1;
            }
        }
        self.lights = lights;
        self.light_count = n as u32;
        self.shadow_light = shadow_idx;

        // Single shadow map for the primary light. HARD RULE on GPU: no light through geometry.
        if shadow_idx >= 0 {
            let bounds = scene.bounds().unwrap_or(forge_core::mesh::Aabb {
                min: glam::Vec3::splat(-5.0),
                max: glam::Vec3::splat(5.0),
            });
            let center = bounds.center();
            let radius = (bounds.radius() + 0.5).max(1.0);

            self.light_view_proj = match shadow_kind {
                LightType::Sun => {
                    let dir = shadow_dir.normalize_or_zero();
                    let up = if dir.y.abs() > 0.9 { glam::Vec3::Z } else { glam::Vec3::Y };
                    let eye = center - dir * (radius * 2.0);
                    let view = Mat4::look_at_rh(eye, center, up);
                    let corners = [
                        glam::Vec3::new(bounds.min.x, bounds.min.y, bounds.min.z),
                        glam::Vec3::new(bounds.max.x, bounds.min.y, bounds.min.z),
                        glam::Vec3::new(bounds.min.x, bounds.max.y, bounds.min.z),
                        glam::Vec3::new(bounds.max.x, bounds.max.y, bounds.min.z),
                        glam::Vec3::new(bounds.min.x, bounds.min.y, bounds.max.z),
                        glam::Vec3::new(bounds.max.x, bounds.min.y, bounds.max.z),
                        glam::Vec3::new(bounds.min.x, bounds.max.y, bounds.max.z),
                        glam::Vec3::new(bounds.max.x, bounds.max.y, bounds.max.z),
                    ];
                    let mut ls_min = glam::Vec3::splat(f32::MAX);
                    let mut ls_max = glam::Vec3::splat(f32::MIN);
                    for c in corners {
                        let p = view.transform_point3(c);
                        ls_min = ls_min.min(p);
                        ls_max = ls_max.max(p);
                    }
                    let pad = (ls_max - ls_min).max(glam::Vec3::splat(0.5)) * 0.08 + glam::Vec3::splat(0.25);
                    ls_min -= pad;
                    ls_max += pad;
                    let z_pad = (ls_max.z - ls_min.z).abs() * 0.25 + 1.0;
                    let near = (-ls_max.z - z_pad).max(0.05);
                    let far = (-ls_min.z + z_pad).max(near + 0.5);
                    let proj = Mat4::orthographic_rh(ls_min.x, ls_max.x, ls_min.y, ls_max.y, near, far);
                    proj * view
                }
                // Point, Spot, Area: perspective from the light toward the scene.
                // Near plane hugs the closest geometry so a lamp between a plane and a cube
                // still records both casters (no light under the cube).
                _ => {
                    let eye = shadow_pos;
                    let target = if shadow_kind == LightType::Spot {
                        eye + shadow_dir.normalize_or_zero() * radius.max(1.0)
                    } else {
                        let to = center - eye;
                        if to.length_squared() < 1e-4 { eye + glam::Vec3::NEG_Y } else { center }
                    };
                    let fwd = (target - eye).normalize_or_zero();
                    let up = if fwd.y.abs() > 0.9 { glam::Vec3::Z } else { glam::Vec3::Y };
                    let view = Mat4::look_at_rh(eye, target, up);
                    let corners = [
                        glam::Vec3::new(bounds.min.x, bounds.min.y, bounds.min.z),
                        glam::Vec3::new(bounds.max.x, bounds.min.y, bounds.min.z),
                        glam::Vec3::new(bounds.min.x, bounds.max.y, bounds.min.z),
                        glam::Vec3::new(bounds.max.x, bounds.max.y, bounds.min.z),
                        glam::Vec3::new(bounds.min.x, bounds.min.y, bounds.max.z),
                        glam::Vec3::new(bounds.max.x, bounds.min.y, bounds.max.z),
                        glam::Vec3::new(bounds.min.x, bounds.max.y, bounds.max.z),
                        glam::Vec3::new(bounds.max.x, bounds.max.y, bounds.max.z),
                    ];
                    let mut max_ang = 0.45f32;
                    let mut min_dist = f32::MAX;
                    let mut max_dist = 0.0f32;
                    for c in corners {
                        let to = c - eye;
                        let d = to.length();
                        if d < 1e-5 { continue; }
                        min_dist = min_dist.min(d);
                        max_dist = max_dist.max(d);
                        max_ang = max_ang.max(fwd.dot(to / d).clamp(-1.0, 1.0).acos());
                    }
                    if min_dist == f32::MAX {
                        min_dist = 0.2;
                        max_dist = radius * 2.0;
                    }
                    let fov = if shadow_kind == LightType::Spot {
                        (shadow_spot_angle.to_radians() * 1.25).clamp(0.25, 2.8)
                    } else {
                        (max_ang * 2.25).clamp(0.8, 2.7)
                    };
                    // Near plane just in front of closest geometry (critical for crevice lights)
                    let near = (min_dist * 0.5).clamp(0.02, 0.5);
                    let far = (max_dist * 1.25 + 0.5).max(near + 1.0);
                    let proj = Mat4::perspective_rh(fov, 1.0, near, far);
                    proj * view
                }
            };
        } else {
            self.light_view_proj = Mat4::IDENTITY;
        }
    }

    fn upload_mesh(&self, m: &MeshData) -> MeshBuffers {
        let edges = build_edges(&m.indices);
        let buf = |label: &str, bytes: &[u8], usage| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytes, usage })
        };
        MeshBuffers {
            vb: buf("vb", bytemuck::cast_slice(&m.vertices), wgpu::BufferUsages::VERTEX),
            ib: buf("ib", bytemuck::cast_slice(&m.indices), wgpu::BufferUsages::INDEX),
            index_count: m.indices.len() as u32,
            wire_ib: buf("wire ib", bytemuck::cast_slice(&edges), wgpu::BufferUsages::INDEX),
            wire_count: edges.len() as u32,
        }
    }

    fn ensure_tex(&self, cache: &mut TexCache, scene: &Scene, idx: usize, srgb: bool) {
        if cache.contains_key(&(idx, srgb)) {
            return;
        }
        let Some(t) = scene.textures.get(idx) else { return };
        if t.width == 0 || t.height == 0 || t.rgba8.len() != t.width as usize * t.height as usize * 4 {
            return;
        }
        let levels = texture::mip_chain_rgba8(t.width, t.height, t.rgba8.clone(), srgb);
        let fmt = if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm };
        cache.insert((idx, srgb), texture::upload_mips(&self.device, &self.queue, &t.name, &levels, fmt));
    }

    fn make_material(&self, m: &Material, scene: &Scene, cache: &mut TexCache) -> GpuMaterial {
        let slots = [
            (m.base_color_tex, true),
            (m.metallic_roughness_tex, false),
            (m.normal_tex, false),
            (m.emissive_tex, true),
            (m.occlusion_tex, false),
        ];
        for (t, srgb) in slots {
            if let Some(i) = t {
                self.ensure_tex(cache, scene, i, srgb);
            }
        }
        fn pick<'a>(
            cache: &'a TexCache,
            t: Option<usize>,
            srgb: bool,
            fallback: &'a wgpu::TextureView,
        ) -> &'a wgpu::TextureView {
            t.and_then(|i| cache.get(&(i, srgb))).unwrap_or(fallback)
        }
        let has_normal = m.normal_tex.map_or(false, |i| cache.contains_key(&(i, false)));
        let blend = m.alpha_mode == AlphaMode::Blend;
        let mut flags = 0u32;
        if has_normal { flags |= 1; }
        if m.alpha_mode == AlphaMode::Mask { flags |= 2; }
        if blend { flags |= 4; }
        let u = MaterialUniform {
            base_color: m.base_color,
            emissive: [m.emissive[0], m.emissive[1], m.emissive[2], m.normal_scale],
            params: [m.metallic, m.roughness, m.alpha_cutoff, m.occlusion_strength],
            flags: [flags, 0, 0, 0],
        };
        let ubuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material uniform"),
            contents: bytemuck::bytes_of(&u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material bg"),
            layout: &self.material_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(pick(cache, m.base_color_tex, true, &self.white)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(pick(cache, m.normal_tex, false, &self.flat_normal)),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(pick(cache, m.metallic_roughness_tex, false, &self.white)),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(pick(cache, m.emissive_tex, true, &self.white)),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(pick(cache, m.occlusion_tex, false, &self.white)),
                },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.mat_sampler) },
            ],
        });
        GpuMaterial { bind_group, blend }
    }

    /// Render the 3D view into `viewport` (x, y, w, h in physical pixels), then let `overlay`
    /// draw UI on top. `overlay` may return extra command buffers to submit first.
    pub fn render(
        &mut self,
        camera: &OrbitCamera,
        s: &ViewSettings,
        viewport: [f32; 4],
        overlay: &mut dyn FnMut(&wgpu::Device, &wgpu::Queue, &mut wgpu::CommandEncoder, &wgpu::TextureView) -> Vec<wgpu::CommandBuffer>,
    ) -> Result<(), wgpu::SurfaceError> {
        let (sw, sh) = (self.config.width as f32, self.config.height as f32);
        let vx = viewport[0].clamp(0.0, (sw - 1.0).max(0.0));
        let vy = viewport[1].clamp(0.0, (sh - 1.0).max(0.0));
        let vw = viewport[2].min(sw - vx).max(1.0);
        let vh = viewport[3].min(sh - vy).max(1.0);

        let vp = camera.view_proj(vw / vh);
        let eye = camera.eye();
        let shadow_on = s.shadows && self.shadow_light >= 0;
        let g = Globals {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            eye: [eye.x, eye.y, eye.z, 1.0],
            env: [s.exposure, s.env_rotation, self.env_max_lod, s.shading as u32 as f32],
            opts: [
                s.show_background as u32 as f32,
                s.background_blur.clamp(0.0, 1.0) * self.env_max_lod,
                s.env_strength,
                if shadow_on { 1.0 + s.shadow_softness } else { 0.0 },
            ],
            sh: self.sh,
            light_info: [self.light_count as f32, self.shadow_light as f32, 0.0, 0.0],
            lights: self.lights,
            light_view_proj: self.light_view_proj.to_cols_array_2d(),
        };
        self.queue.write_buffer(&self.global_buf, 0, bytemuck::bytes_of(&g));

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        // ---- Shadow pass (depth from light) ----
        // HARD RULE: depth from the primary light — geometry blocks light.
        if shadow_on {
            let mut spass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            spass.set_pipeline(&self.shadow_pipeline);
            spass.set_bind_group(0, &self.shadow_global_bg, &[]);
            for o in &self.objects {
                if o.draws.is_empty() || o.buffers.index_count == 0 {
                    continue;
                }
                spass.set_bind_group(1, &o.model_bg, &[]);
                spass.set_vertex_buffer(0, o.buffers.vb.slice(..));
                spass.set_index_buffer(o.buffers.ib.slice(..), wgpu::IndexFormat::Uint32);
                for d in &o.draws {
                    if d.count == 0 {
                        continue;
                    }
                    spass.draw_indexed(d.first..d.first + d.count, 0, 0..1);
                }
            }
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.025, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_viewport(vx, vy, vw, vh, 0.0, 1.0);
            pass.set_scissor_rect(vx as u32, vy as u32, vw as u32, vh as u32);
            pass.set_bind_group(0, &self.global_bg, &[]);

            if s.show_background {
                pass.set_pipeline(&self.sky_pipeline);
                pass.draw(0..3, 0..1);
            }

            pass.set_pipeline(&self.grid_pipeline);
            pass.set_vertex_buffer(0, self.grid_vb.slice(..));
            pass.draw(0..self.grid_count, 0..1);

            if s.wireframe != WireMode::Only {
                for blend in [false, true] {
                    pass.set_pipeline(if blend { &self.mesh_blend } else { &self.mesh_opaque });
                    for o in &self.objects {
                        let mut bound = false;
                        for d in o.draws.iter().filter(|d| d.blend == blend) {
                            if !bound {
                                pass.set_bind_group(2, &o.model_bg, &[]);
                                pass.set_vertex_buffer(0, o.buffers.vb.slice(..));
                                pass.set_index_buffer(o.buffers.ib.slice(..), wgpu::IndexFormat::Uint32);
                                bound = true;
                            }
                            pass.set_bind_group(1, &self.materials[d.material].bind_group, &[]);
                            pass.draw_indexed(d.first..d.first + d.count, 0, 0..1);
                        }
                    }
                }
            }

            // wireframe for everything (if enabled) and always for selected objects
            pass.set_pipeline(&self.wire_pipeline);
            for o in self.objects.iter().filter(|o| o.highlight != Highlight::None || s.wireframe != WireMode::Off) {
                pass.set_bind_group(1, &o.model_bg, &[]);
                pass.set_vertex_buffer(0, o.buffers.vb.slice(..));
                pass.set_index_buffer(o.buffers.wire_ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..o.buffers.wire_count, 0, 0..1);
            }
        }
        let extra = overlay(&self.device, &self.queue, &mut enc, &view);
        self.queue.submit(extra.into_iter().chain(std::iter::once(enc.finish())));
        frame.present();
        Ok(())
    }

    /// Render the scene off-screen at the given resolution and return RGBA8 pixels.
    /// Does not draw UI overlays or selection highlights.

    /// Fast path: only rewrite model matrices (animation scrub / playback).
    /// Call this instead of `set_scene` when meshes/materials did not change.
    pub fn update_transforms(&mut self, scene: &Scene, highlight: &[Highlight]) {
        let worlds = scene.world_matrices();
        let n = self.objects.len().min(scene.objects.len()).min(worlds.len());
        for i in 0..n {
            let hl = highlight.get(i).copied().unwrap_or(Highlight::None);
            let mu = model_uniform(&worlds[i], hl);
            self.queue.write_buffer(&self.objects[i].model_buf, 0, bytemuck::bytes_of(&mu));
            self.objects[i].highlight = hl;
        }
    }

    pub fn capture(
        &mut self,
        camera: &OrbitCamera,
        s: &ViewSettings,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>> {
        self.capture_with(camera, s, width, height, false)
    }

    /// Offline capture. `fast` skips the shadow pass (much quicker for animation).
    pub fn capture_with(
        &mut self,
        camera: &OrbitCamera,
        s: &ViewSettings,
        width: u32,
        height: u32,
        fast: bool,
    ) -> Result<Vec<u8>> {
        let width = width.max(64);
        let height = height.max(64);
        let format = self.config.format;

        // Reuse offscreen targets across frames when size matches.
        let bytes_per_pixel = 4u32;
        let unpadded_bpr = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = (unpadded_bpr + align - 1) / align * align;
        let need_new = match &self.capture_cache {
            Some(c) => c.width != width || c.height != height || c.padded_bpr != padded_bpr,
            None => true,
        };
        if need_new {
            let color_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("capture color"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_view = create_depth(&self.device, width, height);
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("capture staging"),
                size: (padded_bpr * height) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.capture_cache = Some(CaptureCache {
                width,
                height,
                color_tex,
                color_view,
                depth_view,
                staging,
                padded_bpr,
            });
        }
        let cache = self.capture_cache.as_ref().unwrap();
        let color_view = &cache.color_view;
        let depth_view = &cache.depth_view;
        let color_tex = &cache.color_tex;
        let padded_bpr = cache.padded_bpr;

        let vp = camera.view_proj(width as f32 / height as f32);
        let eye = camera.eye();
        let shadow_on = !fast && s.shadows && self.shadow_light >= 0;
        let g = Globals {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            eye: [eye.x, eye.y, eye.z, 1.0],
            env: [s.exposure, s.env_rotation, self.env_max_lod, s.shading as u32 as f32],
            opts: [
                s.show_background as u32 as f32,
                s.background_blur.clamp(0.0, 1.0) * self.env_max_lod,
                s.env_strength,
                if shadow_on { 1.0 + s.shadow_softness } else { 0.0 },
            ],
            sh: self.sh,
            light_info: [self.light_count as f32, self.shadow_light as f32, 0.0, 0.0],
            lights: self.lights,
            light_view_proj: self.light_view_proj.to_cols_array_2d(),
        };
        self.queue.write_buffer(&self.global_buf, 0, bytemuck::bytes_of(&g));

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("capture"),
        });

        if shadow_on {
            let mut spass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("capture shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            spass.set_pipeline(&self.shadow_pipeline);
            spass.set_bind_group(0, &self.shadow_global_bg, &[]);
            for o in &self.objects {
                spass.set_bind_group(1, &o.model_bg, &[]);
                spass.set_vertex_buffer(0, o.buffers.vb.slice(..));
                spass.set_index_buffer(o.buffers.ib.slice(..), wgpu::IndexFormat::Uint32);
                for d in &o.draws {
                    spass.draw_indexed(d.first..d.first + d.count, 0, 0..1);
                }
            }
        }

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("capture main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.025, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, width, height);
            pass.set_bind_group(0, &self.global_bg, &[]);

            if s.show_background {
                pass.set_pipeline(&self.sky_pipeline);
                pass.draw(0..3, 0..1);
            }
            // No ground grid in offline renders.

            if s.wireframe != WireMode::Only {
                for blend in [false, true] {
                    pass.set_pipeline(if blend { &self.mesh_blend } else { &self.mesh_opaque });
                    for o in &self.objects {
                        let mut bound = false;
                        for d in o.draws.iter().filter(|d| d.blend == blend) {
                            if !bound {
                                pass.set_bind_group(2, &o.model_bg, &[]);
                                pass.set_vertex_buffer(0, o.buffers.vb.slice(..));
                                pass.set_index_buffer(o.buffers.ib.slice(..), wgpu::IndexFormat::Uint32);
                                bound = true;
                            }
                            pass.set_bind_group(1, &self.materials[d.material].bind_group, &[]);
                            pass.draw_indexed(d.first..d.first + d.count, 0, 0..1);
                        }
                    }
                }
            }
        }

        // Copy to reused staging buffer
        let unpadded_bpr = width * 4;
        {
            let cache = self.capture_cache.as_ref().unwrap();
            enc.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &cache.color_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &cache.staging,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(cache.padded_bpr),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            );
        }
        self.queue.submit(std::iter::once(enc.finish()));

        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        {
            let cache = self.capture_cache.as_ref().unwrap();
            let slice = cache.staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv().map_err(|e| anyhow!("map channel: {e}"))??;
            {
                let data = slice.get_mapped_range();
                for row in 0..height {
                    let start = (row * padded_bpr) as usize;
                    rgba.extend_from_slice(&data[start..start + unpadded_bpr as usize]);
                }
            } // mapped range dropped
            cache.staging.unmap();
        }

        // Surface format may be Bgra8UnormSrgb — convert to RGBA if needed
        if format == wgpu::TextureFormat::Bgra8Unorm || format == wgpu::TextureFormat::Bgra8UnormSrgb {
            for px in rgba.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        Ok(rgba)
    }
}

fn model_uniform(world: &Mat4, hl: Highlight) -> ModelUniform {
    let m3 = Mat3::from_mat4(*world);
    let det = m3.determinant();
    let nm = if det.abs() > 1e-12 { Mat4::from_mat3(m3.inverse().transpose()) } else { Mat4::IDENTITY };
    let c = match hl {
        Highlight::Active => [1.0, 0.75, 0.2],
        Highlight::Selected => [0.9, 0.35, 0.0],
        Highlight::None => [0.05, 0.05, 0.07],
    };
    ModelUniform {
        matrix: world.to_cols_array_2d(),
        normal_matrix: nm.to_cols_array_2d(),
        tint: [c[0], c[1], c[2], if det < 0.0 { -1.0 } else { 1.0 }],
    }
}

fn make_global_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    buf: &wgpu::Buffer,
    env_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    shadow_view: &wgpu::TextureView,
    shadow_samp: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("global bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(env_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(shadow_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(shadow_samp) },
        ],
    })
}

/// Uploads the env mip chain (RGBA16F) and computes SH diffuse coefficients.
/// Returns (view, sh coefficients, max lod).
fn build_environment(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    env: &EnvironmentMap,
) -> (wgpu::TextureView, [[f32; 4]; 9], f32) {
    let max_w = 2048u32.min(device.limits().max_texture_dimension_2d);
    let mut base = env.clone();
    while base.width > max_w || base.height > max_w {
        base = base.downsample();
    }
    let chain = base.mip_chain();
    let sh_src = chain.iter().find(|m| m.width <= 256).unwrap_or_else(|| chain.last().unwrap());
    let sh9 = sh_src.sh9();
    let mut sh = [[0.0f32; 4]; 9];
    for (o, c) in sh.iter_mut().zip(sh9.iter()) {
        *o = [c[0], c[1], c[2], 0.0];
    }

    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("environment"),
        size: wgpu::Extent3d { width: chain[0].width, height: chain[0].height, depth_or_array_layers: 1 },
        mip_level_count: chain.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ENV_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (i, m) in chain.iter().enumerate() {
        let mut bytes = Vec::with_capacity((m.width * m.height * 8) as usize);
        for px in m.rgb.chunks_exact(3) {
            for c in [px[0], px[1], px[2], 1.0] {
                let c = if c.is_finite() { c.clamp(0.0, 65000.0) } else { 0.0 };
                bytes.extend_from_slice(&half::f16::from_f32(c).to_bits().to_le_bytes());
            }
        }
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: i as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(8 * m.width), rows_per_image: Some(m.height) },
            wgpu::Extent3d { width: m.width, height: m.height, depth_or_array_layers: 1 },
        );
    }
    (tex.create_view(&wgpu::TextureViewDescriptor::default()), sh, (chain.len() - 1) as f32)
}

fn create_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

#[allow(clippy::too_many_arguments)]
fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    wgsl: &str,
    buffers: &[wgpu::VertexBufferLayout],
    topology: wgpu::PrimitiveTopology,
    depth_write: bool,
    depth_compare: wgpu::CompareFunction,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: "vs_main",
            compilation_options: Default::default(),
            buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: "fs_main",
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend: Some(blend), write_mask: wgpu::ColorWrites::ALL })],
        }),
        primitive: wgpu::PrimitiveState { topology, cull_mode: None, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: depth_write,
            depth_compare,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

/// Unique triangle edges as a line-list index buffer (portable wireframe, no POLYGON_MODE_LINE needed).
fn build_edges(indices: &[u32]) -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for t in indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            if seen.insert(key) {
                out.push(a);
                out.push(b);
            }
        }
    }
    out
}

/// Ground grid on the XZ plane (Y-up). X axis red, Z axis blue.
fn build_grid(half: i32, step: f32) -> Vec<LineVertex> {
    let mut v = Vec::new();
    let extent = half as f32 * step;
    for i in -half..=half {
        let t = i as f32 * step;
        let major = i % 5 == 0;
        let c = if major { [0.28, 0.28, 0.31] } else { [0.14, 0.14, 0.16] };
        let cx = if i == 0 { [0.75, 0.2, 0.2] } else { c };
        let cz = if i == 0 { [0.2, 0.35, 0.8] } else { c };
        v.push(LineVertex { pos: [-extent, 0.0, t], color: cx });
        v.push(LineVertex { pos: [extent, 0.0, t], color: cx });
        v.push(LineVertex { pos: [t, 0.0, -extent], color: cz });
        v.push(LineVertex { pos: [t, 0.0, extent], color: cz });
    }
    v
}
