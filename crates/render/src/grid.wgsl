struct VsIn { @location(0) pos: vec3<f32>, @location(1) color: vec3<f32> };
struct VsOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec3<f32> };

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var o: VsOut;
    o.clip = g.view_proj * vec4<f32>(v.pos, 1.0);
    o.color = v.color;
    return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(i.color, 1.0);
}
