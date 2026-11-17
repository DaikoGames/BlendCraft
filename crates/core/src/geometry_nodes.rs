//! Geometry Nodes — Blender-inspired node graph for mesh generation / deformation.
//! Socket colours and layout mirror the Shader Editor conventions used in the app.

use crate::mesh::{MeshData, Vertex};
use crate::primitives::Primitive;
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeoSocketType {
    Geometry,
    Float,
    Vector,
    Int,
    Boolean,
}

#[derive(Clone, Copy, Debug)]
pub struct GeoSocketDef {
    pub name: &'static str,
    pub ty: GeoSocketType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeoNodeKind {
    GroupInput,
    GroupOutput,
    Cube,
    Grid,
    UVSphere,
    IcoSphere,
    Cylinder,
    Cone,
    TransformGeometry,
    JoinGeometry,
    SetPosition,
    MergeByDistance,
    FlipFaces,
    SubdivisionSurface,
    Value,
    Vector,
    Math,
}

impl GeoNodeKind {
    pub fn title(self) -> &'static str {
        match self {
            Self::GroupInput => "Group Input",
            Self::GroupOutput => "Group Output",
            Self::Cube => "Cube",
            Self::Grid => "Grid",
            Self::UVSphere => "UV Sphere",
            Self::IcoSphere => "Ico Sphere",
            Self::Cylinder => "Cylinder",
            Self::Cone => "Cone",
            Self::TransformGeometry => "Transform Geometry",
            Self::JoinGeometry => "Join Geometry",
            Self::SetPosition => "Set Position",
            Self::MergeByDistance => "Merge by Distance",
            Self::FlipFaces => "Flip Faces",
            Self::SubdivisionSurface => "Subdivision Surface",
            Self::Value => "Value",
            Self::Vector => "Vector",
            Self::Math => "Math",
        }
    }

    pub fn category(self) -> &'static str {
        match self {
            Self::GroupInput | Self::GroupOutput => "Input/Output",
            Self::Cube
            | Self::Grid
            | Self::UVSphere
            | Self::IcoSphere
            | Self::Cylinder
            | Self::Cone => "Mesh Primitives",
            Self::TransformGeometry
            | Self::JoinGeometry
            | Self::SetPosition
            | Self::MergeByDistance
            | Self::FlipFaces
            | Self::SubdivisionSurface => "Geometry",
            Self::Value | Self::Vector | Self::Math => "Utilities",
        }
    }

    pub fn all_kinds() -> &'static [GeoNodeKind] {
        &[
            Self::GroupInput,
            Self::GroupOutput,
            Self::Cube,
            Self::Grid,
            Self::UVSphere,
            Self::IcoSphere,
            Self::Cylinder,
            Self::Cone,
            Self::TransformGeometry,
            Self::JoinGeometry,
            Self::SetPosition,
            Self::MergeByDistance,
            Self::FlipFaces,
            Self::SubdivisionSurface,
            Self::Value,
            Self::Vector,
            Self::Math,
        ]
    }

    pub fn inputs(self) -> &'static [GeoSocketDef] {
        use GeoSocketType::*;
        match self {
            Self::GroupInput | Self::Cube | Self::Grid | Self::UVSphere | Self::IcoSphere
            | Self::Cylinder | Self::Cone | Self::Value | Self::Vector => &[],
            Self::GroupOutput | Self::FlipFaces | Self::SubdivisionSurface => &[GeoSocketDef {
                name: "Geometry",
                ty: Geometry,
            }],
            Self::TransformGeometry => &[
                GeoSocketDef {
                    name: "Geometry",
                    ty: Geometry,
                },
                GeoSocketDef {
                    name: "Translation",
                    ty: Vector,
                },
                GeoSocketDef {
                    name: "Rotation",
                    ty: Vector,
                },
                GeoSocketDef {
                    name: "Scale",
                    ty: Vector,
                },
            ],
            Self::JoinGeometry => &[
                GeoSocketDef {
                    name: "Geometry",
                    ty: Geometry,
                },
                GeoSocketDef {
                    name: "Geometry",
                    ty: Geometry,
                },
            ],
            Self::SetPosition => &[
                GeoSocketDef {
                    name: "Geometry",
                    ty: Geometry,
                },
                GeoSocketDef {
                    name: "Offset",
                    ty: Vector,
                },
            ],
            Self::MergeByDistance => &[
                GeoSocketDef {
                    name: "Geometry",
                    ty: Geometry,
                },
                GeoSocketDef {
                    name: "Distance",
                    ty: Float,
                },
            ],
            Self::Math => &[
                GeoSocketDef {
                    name: "Value",
                    ty: Float,
                },
                GeoSocketDef {
                    name: "Value",
                    ty: Float,
                },
            ],
        }
    }

    pub fn outputs(self) -> &'static [GeoSocketDef] {
        use GeoSocketType::*;
        match self {
            Self::GroupOutput => &[],
            Self::Value | Self::Math => &[GeoSocketDef {
                name: "Value",
                ty: Float,
            }],
            Self::Vector => &[GeoSocketDef {
                name: "Vector",
                ty: Vector,
            }],
            _ => &[GeoSocketDef {
                name: "Geometry",
                ty: Geometry,
            }],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeoNode {
    pub id: u64,
    pub kind: GeoNodeKind,
    pub pos: [f32; 2],
    pub value: f32,
    pub secondary: f32,
    pub tertiary: f32,
    pub vec: [f32; 3],
    pub vec2: [f32; 3],
}

impl GeoNode {
    pub fn new(id: u64, kind: GeoNodeKind, pos: [f32; 2]) -> Self {
        let (value, secondary, tertiary, vec, vec2) = default_params(kind);
        Self {
            id,
            kind,
            pos,
            value,
            secondary,
            tertiary,
            vec,
            vec2,
        }
    }
}

fn default_params(kind: GeoNodeKind) -> (f32, f32, f32, [f32; 3], [f32; 3]) {
    match kind {
        GeoNodeKind::Cube => (1.0, 0.0, 0.0, [0.0; 3], [0.0; 3]),
        GeoNodeKind::Grid => (2.0, 10.0, 10.0, [0.0; 3], [0.0; 3]),
        GeoNodeKind::UVSphere => (0.5, 32.0, 16.0, [0.0; 3], [0.0; 3]),
        GeoNodeKind::IcoSphere => (0.5, 2.0, 0.0, [0.0; 3], [0.0; 3]),
        GeoNodeKind::Cylinder => (0.5, 1.0, 32.0, [0.0; 3], [0.0; 3]),
        GeoNodeKind::Cone => (0.5, 1.0, 32.0, [0.0; 3], [0.0; 3]),
        GeoNodeKind::TransformGeometry => (0.0, 0.0, 0.0, [0.0; 3], [1.0, 1.0, 1.0]),
        GeoNodeKind::SetPosition => (0.0, 0.0, 0.0, [0.0, 0.1, 0.0], [0.0; 3]),
        GeoNodeKind::MergeByDistance => (0.001, 0.0, 0.0, [0.0; 3], [0.0; 3]),
        GeoNodeKind::SubdivisionSurface => (1.0, 0.0, 0.0, [0.0; 3], [0.0; 3]),
        GeoNodeKind::Value => (1.0, 0.0, 0.0, [0.0; 3], [0.0; 3]),
        GeoNodeKind::Vector => (0.0, 0.0, 0.0, [0.0; 3], [0.0; 3]),
        GeoNodeKind::Math => (0.0, 0.0, 0.0, [0.0; 3], [0.0; 3]),
        _ => (0.0, 0.0, 0.0, [0.0; 3], [0.0; 3]),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeoLink {
    pub from: u64,
    pub from_socket: u8,
    pub to: u64,
    pub to_socket: u8,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeometryNodeGraph {
    pub nodes: Vec<GeoNode>,
    pub links: Vec<GeoLink>,
    pub next_id: u64,
    pub enabled: bool,
}

impl Default for GeometryNodeGraph {
    fn default() -> Self {
        Self {
            nodes: vec![
                GeoNode::new(1, GeoNodeKind::GroupInput, [40.0, 120.0]),
                GeoNode::new(2, GeoNodeKind::GroupOutput, [520.0, 120.0]),
            ],
            links: vec![GeoLink {
                from: 1,
                from_socket: 0,
                to: 2,
                to_socket: 0,
            }],
            next_id: 3,
            enabled: true,
        }
    }
}

impl GeometryNodeGraph {
    pub fn add_node(&mut self, kind: GeoNodeKind, pos: [f32; 2]) -> u64 {
        let id = self.next_id.max(1);
        self.next_id = id.saturating_add(1);
        self.nodes.push(GeoNode::new(id, kind, pos));
        id
    }

    pub fn remove_node(&mut self, id: u64) {
        if matches!(
            self.nodes.iter().find(|n| n.id == id).map(|n| n.kind),
            Some(GeoNodeKind::GroupInput | GeoNodeKind::GroupOutput)
        ) {
            return;
        }
        self.nodes.retain(|n| n.id != id);
        self.links.retain(|l| l.from != id && l.to != id);
    }

    pub fn connect(&mut self, from: u64, from_socket: u8, to: u64, to_socket: u8) -> bool {
        if from == to {
            return false;
        }
        let Some(from_n) = self.nodes.iter().find(|n| n.id == from) else {
            return false;
        };
        let Some(to_n) = self.nodes.iter().find(|n| n.id == to) else {
            return false;
        };
        let outs = from_n.kind.outputs();
        let ins = to_n.kind.inputs();
        if from_socket as usize >= outs.len() || to_socket as usize >= ins.len() {
            return false;
        }
        if outs[from_socket as usize].ty != ins[to_socket as usize].ty {
            return false;
        }
        self.links
            .retain(|l| !(l.to == to && l.to_socket == to_socket));
        self.links.push(GeoLink {
            from,
            from_socket,
            to,
            to_socket,
        });
        if self.topo_order().is_none() {
            self.links.pop();
            return false;
        }
        true
    }

    fn topo_order(&self) -> Option<Vec<u64>> {
        let mut indeg: HashMap<u64, usize> = HashMap::new();
        for n in &self.nodes {
            indeg.insert(n.id, 0);
        }
        for l in &self.links {
            *indeg.entry(l.to).or_insert(0) += 1;
        }
        let mut q: VecDeque<u64> = indeg
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| *id)
            .collect();
        let mut out = Vec::with_capacity(self.nodes.len());
        while let Some(id) = q.pop_front() {
            out.push(id);
            for l in self.links.iter().filter(|l| l.from == id) {
                if let Some(d) = indeg.get_mut(&l.to) {
                    *d = d.saturating_sub(1);
                    if *d == 0 {
                        q.push_back(l.to);
                    }
                }
            }
        }
        (out.len() == self.nodes.len()).then_some(out)
    }

    /// Evaluate the graph. `input` is the object's original mesh (Group Input).
    pub fn evaluate(&self, input: &MeshData) -> Option<MeshData> {
        if !self.enabled {
            return Some(input.clone());
        }
        let order = self.topo_order()?;
        let mut geo: HashMap<u64, MeshData> = HashMap::new();
        let mut floats: HashMap<(u64, u8), f32> = HashMap::new();
        let mut vectors: HashMap<(u64, u8), [f32; 3]> = HashMap::new();

        for id in order {
            let node = self.nodes.iter().find(|n| n.id == id)?;
            match node.kind {
                GeoNodeKind::GroupInput => {
                    geo.insert(id, input.clone());
                }
                GeoNodeKind::GroupOutput => {}
                GeoNodeKind::Cube => {
                    geo.insert(id, Primitive::Cube { size: node.value.max(0.01) }.build());
                }
                GeoNodeKind::Grid => {
                    let size = node.value.max(0.01);
                    let xd = node.secondary.round().clamp(1.0, 64.0) as u32;
                    let yd = node.tertiary.round().clamp(1.0, 64.0) as u32;
                    geo.insert(id, Primitive::Grid { size, x_div: xd, y_div: yd }.build());
                }
                GeoNodeKind::UVSphere => {
                    geo.insert(
                        id,
                        Primitive::Sphere {
                            radius: node.value.max(0.01),
                            segments: node.secondary.round().clamp(3.0, 64.0) as u32,
                            rings: node.tertiary.round().clamp(2.0, 64.0) as u32,
                        }
                        .build(),
                    );
                }
                GeoNodeKind::IcoSphere => {
                    geo.insert(
                        id,
                        Primitive::IcoSphere {
                            radius: node.value.max(0.01),
                            subdivisions: node.secondary.round().clamp(0.0, 5.0) as u32,
                        }
                        .build(),
                    );
                }
                GeoNodeKind::Cylinder => {
                    geo.insert(
                        id,
                        Primitive::Cylinder {
                            radius: node.value.max(0.01),
                            height: node.secondary.max(0.01),
                            segments: node.tertiary.round().clamp(3.0, 64.0) as u32,
                        }
                        .build(),
                    );
                }
                GeoNodeKind::Cone => {
                    geo.insert(
                        id,
                        Primitive::Cone {
                            radius: node.value.max(0.01),
                            height: node.secondary.max(0.01),
                            segments: node.tertiary.round().clamp(3.0, 64.0) as u32,
                        }
                        .build(),
                    );
                }
                GeoNodeKind::TransformGeometry => {
                    let mesh = self
                        .linked_geo(id, 0, &geo, input)
                        .unwrap_or_else(|| input.clone());
                    let t = self.linked_vec(id, 1, &vectors).unwrap_or(node.vec);
                    let rot_deg = self.linked_vec(id, 2, &vectors).unwrap_or([0.0; 3]);
                    let scale = self.linked_vec(id, 3, &vectors).unwrap_or(node.vec2);
                    geo.insert(id, transform_mesh(&mesh, t, rot_deg, scale));
                }
                GeoNodeKind::JoinGeometry => {
                    let a = self
                        .linked_geo(id, 0, &geo, input)
                        .unwrap_or_else(MeshData::default);
                    let b = self
                        .linked_geo(id, 1, &geo, input)
                        .unwrap_or_else(MeshData::default);
                    geo.insert(id, join_meshes(&a, &b));
                }
                GeoNodeKind::SetPosition => {
                    let mesh = self
                        .linked_geo(id, 0, &geo, input)
                        .unwrap_or_else(|| input.clone());
                    let offset = self.linked_vec(id, 1, &vectors).unwrap_or(node.vec);
                    geo.insert(id, offset_mesh(&mesh, offset));
                }
                GeoNodeKind::MergeByDistance => {
                    let mesh = self
                        .linked_geo(id, 0, &geo, input)
                        .unwrap_or_else(|| input.clone());
                    let dist = self
                        .linked_float(id, 1, &floats)
                        .unwrap_or(node.value)
                        .max(0.0);
                    geo.insert(id, merge_by_distance(&mesh, dist));
                }
                GeoNodeKind::FlipFaces => {
                    let mesh = self
                        .linked_geo(id, 0, &geo, input)
                        .unwrap_or_else(|| input.clone());
                    geo.insert(id, flip_faces(&mesh));
                }
                GeoNodeKind::SubdivisionSurface => {
                    let mesh = self
                        .linked_geo(id, 0, &geo, input)
                        .unwrap_or_else(|| input.clone());
                    let levels = node.value.round().clamp(0.0, 2.0) as u32;
                    geo.insert(id, subdivide_simple(&mesh, levels));
                }
                GeoNodeKind::Value => {
                    floats.insert((id, 0), node.value);
                }
                GeoNodeKind::Vector => {
                    vectors.insert((id, 0), node.vec);
                }
                GeoNodeKind::Math => {
                    let a = self.linked_float(id, 0, &floats).unwrap_or(node.value);
                    let b = self.linked_float(id, 1, &floats).unwrap_or(node.secondary);
                    let r = match node.tertiary.round() as i32 {
                        1 => a - b,
                        2 => a * b,
                        3 => {
                            if b.abs() < 1e-8 {
                                0.0
                            } else {
                                a / b
                            }
                        }
                        _ => a + b,
                    };
                    floats.insert((id, 0), r);
                }
            }
        }

        let out_id = self
            .nodes
            .iter()
            .find(|n| n.kind == GeoNodeKind::GroupOutput)
            .map(|n| n.id)?;
        self.linked_geo(out_id, 0, &geo, input)
    }

    fn linked_geo(
        &self,
        to: u64,
        socket: u8,
        geo: &HashMap<u64, MeshData>,
        fallback: &MeshData,
    ) -> Option<MeshData> {
        let link = self
            .links
            .iter()
            .find(|l| l.to == to && l.to_socket == socket)?;
        geo.get(&link.from)
            .cloned()
            .or_else(|| Some(fallback.clone()))
    }

    fn linked_float(&self, to: u64, socket: u8, floats: &HashMap<(u64, u8), f32>) -> Option<f32> {
        let link = self
            .links
            .iter()
            .find(|l| l.to == to && l.to_socket == socket)?;
        floats
            .get(&(link.from, link.from_socket))
            .copied()
            .or_else(|| floats.get(&(link.from, 0)).copied())
    }

    fn linked_vec(
        &self,
        to: u64,
        socket: u8,
        vectors: &HashMap<(u64, u8), [f32; 3]>,
    ) -> Option<[f32; 3]> {
        let link = self
            .links
            .iter()
            .find(|l| l.to == to && l.to_socket == socket)?;
        vectors
            .get(&(link.from, link.from_socket))
            .copied()
            .or_else(|| vectors.get(&(link.from, 0)).copied())
    }
}

fn transform_mesh(
    mesh: &MeshData,
    translation: [f32; 3],
    rot_deg: [f32; 3],
    scale: [f32; 3],
) -> MeshData {
    let t = Vec3::from(translation);
    let r = Quat::from_euler(
        glam::EulerRot::XYZ,
        rot_deg[0].to_radians(),
        rot_deg[1].to_radians(),
        rot_deg[2].to_radians(),
    );
    let s = Vec3::from(scale);
    let m = Mat4::from_scale_rotation_translation(s, r, t);
    mesh.transformed(m)
}

fn offset_mesh(mesh: &MeshData, offset: [f32; 3]) -> MeshData {
    let o = Vec3::from(offset);
    let mut out = mesh.clone();
    for v in &mut out.vertices {
        v.position = (Vec3::from(v.position) + o).to_array();
    }
    out
}

fn join_meshes(a: &MeshData, b: &MeshData) -> MeshData {
    let mut out = a.clone();
    let base = out.vertices.len() as u32;
    out.vertices.extend_from_slice(&b.vertices);
    out.indices
        .extend(b.indices.iter().map(|i| i.saturating_add(base)));
    out
}

fn flip_faces(mesh: &MeshData) -> MeshData {
    let mut out = mesh.clone();
    for tri in out.indices.chunks_exact_mut(3) {
        tri.swap(1, 2);
    }
    for v in &mut out.vertices {
        v.normal = [
            -v.normal[0],
            -v.normal[1],
            -v.normal[2],
        ];
    }
    out
}

fn merge_by_distance(mesh: &MeshData, dist: f32) -> MeshData {
    if dist <= 0.0 || mesh.vertices.is_empty() {
        return mesh.clone();
    }
    let n = mesh.vertices.len();
    let mut map = vec![0u32; n];
    let mut kept: Vec<usize> = Vec::new();
    let d2 = dist * dist;
    for i in 0..n {
        let pi = Vec3::from(mesh.vertices[i].position);
        let mut found = None;
        for &k in &kept {
            let pk = Vec3::from(mesh.vertices[k].position);
            if (pi - pk).length_squared() <= d2 {
                found = Some(k);
                break;
            }
        }
        if let Some(k) = found {
            map[i] = kept.iter().position(|&x| x == k).unwrap() as u32;
        } else {
            map[i] = kept.len() as u32;
            kept.push(i);
        }
    }
    let vertices: Vec<Vertex> = kept.iter().map(|&i| mesh.vertices[i]).collect();
    let mut indices: Vec<u32> = mesh
        .indices
        .iter()
        .map(|&i| map.get(i as usize).copied().unwrap_or(0))
        .collect();
    let mut clean = Vec::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
            clean.extend_from_slice(tri);
        }
    }
    indices = clean;
    MeshData {
        name: mesh.name.clone(),
        vertices,
        indices,
        groups: vec![],
    }
}

fn subdivide_simple(mesh: &MeshData, levels: u32) -> MeshData {
    let mut current = mesh.clone();
    for _ in 0..levels {
        current = subdivide_once(&current);
    }
    current
}

fn subdivide_once(mesh: &MeshData) -> MeshData {
    let mut vertices = mesh.vertices.clone();
    let mut edge_mid: HashMap<(u32, u32), u32> = HashMap::new();
    let mut mid = |a: u32, b: u32, vertices: &mut Vec<Vertex>| -> u32 {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&m) = edge_mid.get(&key) {
            return m;
        }
        let va = vertices[a as usize];
        let vb = vertices[b as usize];
        let pm = (Vec3::from(va.position) + Vec3::from(vb.position)) * 0.5;
        let nm = (Vec3::from(va.normal) + Vec3::from(vb.normal)).normalize_or_zero();
        let um = [(va.uv[0] + vb.uv[0]) * 0.5, (va.uv[1] + vb.uv[1]) * 0.5];
        let idx = vertices.len() as u32;
        vertices.push(Vertex {
            position: pm.to_array(),
            normal: nm.to_array(),
            uv: um,
        });
        edge_mid.insert(key, idx);
        idx
    };
    let mut indices = Vec::with_capacity(mesh.indices.len() * 4);
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        let ab = mid(a, b, &mut vertices);
        let bc = mid(b, c, &mut vertices);
        let ca = mid(c, a, &mut vertices);
        indices.extend_from_slice(&[a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca]);
    }
    MeshData {
        name: mesh.name.clone(),
        vertices,
        indices,
        groups: vec![],
    }
}
