struct Model {
    matrix: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    tint: vec4<f32>,
};
@group(1) @binding(0) var<uniform> model: Model;

// Depth-only pass used by Wireframe mode when X-Ray is OFF: the surface writes depth
// (but no colour) so edges behind it are hidden instead of showing through.
@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return g.view_proj * (model.matrix * vec4<f32>(pos, 1.0));
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
