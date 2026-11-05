struct Material {
    base_color: vec4<f32>,
    emissive: vec4<f32>,   // rgb = emissive factor, w = normal scale
    params: vec4<f32>,     // x metallic, y roughness, z alpha cutoff, w occlusion strength
    flags: vec4<u32>,      // x: bit0 has_normal_map, bit1 alpha_mask, bit2 alpha_blend
};
@group(1) @binding(0) var<uniform> u_mat: Material;
@group(1) @binding(1) var base_tex: texture_2d<f32>;
@group(1) @binding(2) var normal_tex: texture_2d<f32>;
@group(1) @binding(3) var mr_tex: texture_2d<f32>;
@group(1) @binding(4) var emissive_tex: texture_2d<f32>;
@group(1) @binding(5) var occlusion_tex: texture_2d<f32>;
@group(1) @binding(6) var mat_samp: sampler;

struct Model {
    matrix: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    tint: vec4<f32>,       // w = handedness (-1 for mirrored transforms)
};
@group(2) @binding(0) var<uniform> model: Model;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var o: VsOut;
    let wp = model.matrix * vec4<f32>(v.pos, 1.0);
    o.clip = g.view_proj * wp;
    o.normal = (model.normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    o.world = wp.xyz;
    o.uv = v.uv;
    return o;
}

// Normal mapping without stored tangents: derive T/B from screen-space derivatives.
// Solved explicitly so the result does not depend on the screen-space Y direction.
fn perturb_normal(n: vec3<f32>, p: vec3<f32>, uv_in: vec2<f32>, tn: vec3<f32>) -> vec3<f32> {
    let uv = vec2<f32>(uv_in.x, 1.0 - uv_in.y); // glTF: v points down, normal map +Y points up
    let dp1 = dpdx(p);
    let dp2 = dpdy(p);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let det = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(det) < 1e-20) {
        return n;
    }
    var t = (dp1 * duv2.y - dp2 * duv1.y) / det;
    var b = (dp2 * duv1.x - dp1 * duv2.x) / det;
    t = t - n * dot(n, t);
    b = b - n * dot(n, b);
    if (dot(t, t) < 1e-30 || dot(b, b) < 1e-30) {
        return n;
    }
    t = normalize(t);
    b = normalize(b);
    return normalize(t * tn.x + b * tn.y + n * tn.z);
}

// Karis' analytic approximation of the split-sum environment BRDF (no LUT texture needed)
fn env_brdf_approx(f0: vec3<f32>, roughness: f32, nov: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nov)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + vec3<f32>(ab.y);
}

fn ggx_direct(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, base: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
    let nl = max(dot(n, l), 0.0);
    if (nl <= 0.0) {
        return vec3<f32>(0.0);
    }
    let nv = max(dot(n, v), 1e-4);
    let h = normalize(v + l);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(v, h), 0.0);
    let a = roughness * roughness;
    let a2 = a * a;
    let d_den = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * d_den * d_den);
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let geo = (nl / (nl * (1.0 - k) + k)) * (nv / (nv * (1.0 - k) + k));
    let f0 = mix(vec3<f32>(0.04), base, metallic);
    let f = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - vh, 5.0);
    let spec = d * geo * f / max(4.0 * nl * nv, 1e-4);
    let diffuse = base * (1.0 - metallic) / PI;
    return (diffuse * (vec3<f32>(1.0) - f) + spec) * nl;
}

fn debug_out(x: vec3<f32>) -> vec3<f32> {
    return pow(x, vec3<f32>(2.2)); // so the sRGB surface shows the authored values
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let base_s = textureSample(base_tex, mat_samp, uv);
    let nrm_s = textureSample(normal_tex, mat_samp, uv).xyz;
    let mr_s = textureSample(mr_tex, mat_samp, uv);
    let em_s = textureSample(emissive_tex, mat_samp, uv).rgb;
    let ao_s = textureSample(occlusion_tex, mat_samp, uv).r;

    let flags = u_mat.flags.x;
    let has_normal = (flags & 1u) != 0u;
    let is_mask = (flags & 2u) != 0u;
    let is_blend = (flags & 4u) != 0u;

    let base = u_mat.base_color * base_s;
    let metallic = clamp(u_mat.params.x * mr_s.b, 0.0, 1.0);
    let roughness = clamp(u_mat.params.y * mr_s.g, 0.045, 1.0);
    let ao = 1.0 + u_mat.params.w * (ao_s - 1.0);

    var n = normalize(in.normal);
    let flip = select(!front, front, model.tint.w < 0.0);
    if (flip) {
        n = -n;
    }
    var tn = nrm_s * 2.0 - 1.0;
    tn = vec3<f32>(tn.xy * u_mat.emissive.w, tn.z);
    let np = perturb_normal(n, in.world, uv, tn);
    n = select(n, np, has_normal);

    if (is_mask && base.a < u_mat.params.z) {
        discard;
    }

    let v = normalize(g.eye.xyz - in.world);
    let nov = max(dot(n, v), 1e-4);
    let f0 = mix(vec3<f32>(0.04), base.rgb, metallic);
    let diffuse_color = base.rgb * (1.0 - metallic);

    let irradiance = sh_irradiance(rot_env(n));
    let rdir = reflect(-v, n);
    let prefiltered = sample_env(rdir, roughness * g.env.z);
    let spec = prefiltered * env_brdf_approx(f0, roughness, nov);

    // scene lights (point / sun / spot), no shadows yet
    var direct = vec3<f32>(0.0);
    let light_count = u32(g.light_info.x + 0.5);
    for (var i: u32 = 0u; i < light_count; i = i + 1u) {
        let lt = g.lights[i];
        var l = vec3<f32>(0.0, 1.0, 0.0);
        var atten = 1.0;
        if (lt.pos_type.w < 0.5 || lt.pos_type.w > 1.5) {
            let dvec = lt.pos_type.xyz - in.world;
            let dist2 = max(dot(dvec, dvec), 1e-4);
            l = dvec * inverseSqrt(dist2);
            atten = 1.0 / dist2;
            if (lt.color.w > 0.0) {
                let r2 = lt.color.w * lt.color.w;
                let f = clamp(1.0 - (dist2 / r2) * (dist2 / r2), 0.0, 1.0);
                atten = atten * f * f;
            }
            if (lt.pos_type.w > 1.5) {
                let cd = dot(-l, lt.dir.xyz);
                atten = atten * smoothstep(lt.spot.x, lt.spot.y, cd);
            }
        } else {
            l = -lt.dir.xyz;
        }
        direct = direct + ggx_direct(n, v, l, base.rgb, metallic, roughness) * lt.color.rgb * atten;
    }

    let lit = (diffuse_color * irradiance + spec) * ao * g.opts.z + direct + em_s * u_mat.emissive.rgb;

    let mode = u32(g.env.w + 0.5);
    var rgb = finish_color(lit);
    if (mode == 1u) {
        rgb = base.rgb;
    } else if (mode == 2u) {
        rgb = debug_out(n * 0.5 + 0.5);
    } else if (mode == 3u) {
        rgb = debug_out(vec3<f32>(roughness));
    } else if (mode == 4u) {
        rgb = debug_out(vec3<f32>(metallic));
    } else if (mode == 5u) {
        let cell = floor(uv * 8.0);
        let chk = fract((cell.x + cell.y) * 0.5) * 2.0;
        let tint = 0.7 + 0.3 * vec3<f32>(uv.x, uv.y, 1.0 - uv.x);
        rgb = mix(vec3<f32>(0.1), vec3<f32>(0.85), chk) * tint;
    }
    return vec4<f32>(rgb, select(1.0, base.a, is_blend));
}
