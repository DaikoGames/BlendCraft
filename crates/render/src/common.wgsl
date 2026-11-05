struct Light {
    pos_type: vec4<f32>,   // xyz position, w type: 0 point, 1 sun, 2 spot
    color: vec4<f32>,      // rgb = color * intensity, w = range (0 = unlimited)
    dir: vec4<f32>,        // xyz = direction the light travels
    spot: vec4<f32>,       // x = cos(outer), y = cos(inner)
};

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    env: vec4<f32>,   // x = exposure, y = env rotation (rad), z = env max lod, w = shading mode
    opts: vec4<f32>,  // x = show background, y = background lod
    sh: array<vec4<f32>, 9>,
    light_info: vec4<f32>, // x = light count
    lights: array<Light, 8>,
};
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var env_tex: texture_2d<f32>;
@group(0) @binding(2) var env_samp: sampler;

const PI: f32 = 3.14159265359;

fn rot_env(d: vec3<f32>) -> vec3<f32> {
    let c = cos(g.env.y);
    let s = sin(g.env.y);
    return vec3<f32>(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
}

// must match EnvironmentMap::texel_dir on the CPU
fn equirect_uv(d: vec3<f32>) -> vec2<f32> {
    let u = atan2(d.z, d.x) / (2.0 * PI) + 0.5;
    let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2<f32>(u, v);
}

fn sample_env(d: vec3<f32>, lod: f32) -> vec3<f32> {
    return textureSampleLevel(env_tex, env_samp, equirect_uv(rot_env(d)), lod).rgb;
}

fn sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    let x = n.x;
    let y = n.y;
    let z = n.z;
    var c = g.sh[0].rgb * 0.282095;
    c += g.sh[1].rgb * (0.488603 * y);
    c += g.sh[2].rgb * (0.488603 * z);
    c += g.sh[3].rgb * (0.488603 * x);
    c += g.sh[4].rgb * (1.092548 * x * y);
    c += g.sh[5].rgb * (1.092548 * y * z);
    c += g.sh[6].rgb * (0.315392 * (3.0 * z * z - 1.0));
    c += g.sh[7].rgb * (1.092548 * x * z);
    c += g.sh[8].rgb * (0.546274 * (x * x - y * y));
    return max(c, vec3<f32>(0.0));
}

fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn finish_color(c: vec3<f32>) -> vec3<f32> {
    return tonemap_aces(c * g.env.x);
}
