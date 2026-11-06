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
    env: vec4<f32>,   // x = exposure, y = env rotation, z = env max lod, w = shading mode
    opts: vec4<f32>,  // x = show bg, y = bg lod, z = env_strength, w = shadow enable + softness
    sh: array<vec4<f32>, 9>,
    light_info: vec4<f32>, // x = count, y = shadow light index
    lights: array<Light, 8>,
    light_view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var env_tex: texture_2d<f32>;
@group(0) @binding(2) var env_samp: sampler;
@group(0) @binding(3) var shadow_map: texture_depth_2d;
@group(0) @binding(4) var shadow_samp: sampler_comparison;

const PI: f32 = 3.14159265359;

fn rot_env(d: vec3<f32>) -> vec3<f32> {
    let c = cos(g.env.y);
    let s = sin(g.env.y);
    return vec3<f32>(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
}

fn equirect_uv(d: vec3<f32>) -> vec2<f32> {
    let u = atan2(d.z, d.x) / (2.0 * PI) + 0.5;
    let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2<f32>(u, v);
}

fn sample_env(dir: vec3<f32>, lod: f32) -> vec3<f32> {
    let d = rot_env(normalize(dir));
    return textureSampleLevel(env_tex, env_samp, equirect_uv(d), lod).rgb;
}

fn sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    let nr = rot_env(n);
    let x = nr.x;
    let y = nr.y;
    let z = nr.z;
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

// UNIVERSAL RULE: light does not pass through geometry (1 = lit, 0 = blocked).
// Soft PCF so round meshes don't show triangular shadow edges.
fn sample_shadow(world: vec3<f32>, n: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let ndotl = max(dot(n, light_dir), 0.0);
    if (ndotl <= 0.0) {
        return 0.0;
    }

    let lp = g.light_view_proj * vec4<f32>(world, 1.0);
    let ndc = lp.xyz / max(abs(lp.w), 1e-6);
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let z = ndc.z;

    let shadow_idx = i32(g.light_info.y);
    var outside = 1.0;
    if (shadow_idx >= 0) {
        let t = g.lights[u32(shadow_idx)].pos_type.w;
        if (t > 1.5) { outside = 0.0; } // spot outside beam
    }
    if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0 || z <= 0.0 || z >= 1.0) {
        return outside;
    }

    // Gentle slope bias — enough against acne, low enough for soft contact
    let bias = 0.0004 + 0.0025 * (1.0 - ndotl);
    let ref_z = clamp(z - bias, 0.0, 0.999);

    // Softness from UI (opts.w = 1 + softness); keep a visible penumbra on curves
    let softness = max(g.opts.w - 1.0, 0.75);
    let texel = 1.0 / 4096.0;
    // Wider kernel = softer silhouette (hides triangle edges on spheres/capsules)
    let radius = (1.25 + softness * 1.75) * texel;

    // 5x5 PCF with gaussian-ish weights
    var sum = 0.0;
    var wsum = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let fx = f32(dx);
            let fy = f32(dy);
            // Center-weighted disc
            let w = exp(-0.35 * (fx * fx + fy * fy));
            let off = vec2<f32>(fx, fy) * radius;
            sum += textureSampleCompare(shadow_map, shadow_samp, uv + off, ref_z) * w;
            wsum += w;
        }
    }
    return sum / wsum;
}
