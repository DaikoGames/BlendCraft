struct Model {
    matrix: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    tint: vec4<f32>,
};
@group(1) @binding(0) var<uniform> model: Model;

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    var c = g.view_proj * (model.matrix * vec4<f32>(pos, 1.0));
    c.z = c.z - 0.00005 * c.w; // pull lines slightly toward the camera so they win over their own surface
    return c;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(model.tint.rgb, 1.0);
}
