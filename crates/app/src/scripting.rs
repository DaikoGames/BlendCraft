use glam::{EulerRot, Quat};
use serde::{Deserialize, Serialize};

pub fn rotation_quaternion(euler: [f32; 3]) -> Quat {
    Quat::from_euler(EulerRot::XYZ, euler[0], euler[1], euler[2])
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScriptObject {
    pub id: Option<u64>,
    pub name: String,
    #[serde(rename = "type")]
    pub object_type: String,
    #[serde(default)]
    pub primitive: String,
    #[serde(default)]
    pub primitive_params: Vec<f32>,
    #[serde(default)]
    pub material: Option<usize>,
    pub location: [f32; 3],
    pub rotation_euler: [f32; 3],
    pub scale: [f32; 3],
    pub selected: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScriptMaterial {
    pub name: String,
    pub diffuse_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScriptSnapshot {
    pub objects: Vec<ScriptObject>,
    #[serde(default)]
    pub materials: Vec<ScriptMaterial>,
    pub active: Option<usize>,
}

#[derive(Debug)]
pub struct ScriptResult {
    pub objects: Vec<ScriptObject>,
    pub materials: Vec<ScriptMaterial>,
    pub active: Option<usize>,
    pub output: String,
}

#[derive(Debug)]
pub struct ScriptError {
    pub output: String,
    pub message: String,
}

#[derive(Debug)]
pub struct ScriptCompletion {
    pub document_version: u64,
    pub result: Result<ScriptResult, ScriptError>,
}

impl ScriptSnapshot {
    pub fn from_document(doc: &forge_core::Document) -> Self {
        let active = doc
            .active
            .and_then(|id| doc.scene.objects.iter().position(|object| object.id == id));
        let objects = doc
            .scene
            .objects
            .iter()
            .map(|object| {
                let (x, y, z) = object.rotation.to_euler(EulerRot::XYZ);
                let (primitive, primitive_params) = object
                    .primitive
                    .as_ref()
                    .map(primitive_script_data)
                    .unwrap_or_else(|| ("NONE".to_owned(), Vec::new()));
                ScriptObject {
                    id: Some(object.id),
                    name: object.name.clone(),
                    object_type: match object.kind {
                        forge_core::ObjectKind::Mesh => "MESH",
                        forge_core::ObjectKind::Camera(_) => "CAMERA",
                        forge_core::ObjectKind::Light(_) => "LIGHT",
                        _ => "OTHER",
                    }
                    .to_owned(),
                    primitive,
                    primitive_params,
                    material: object.material,
                    location: object.translation.to_array(),
                    rotation_euler: [x, y, z],
                    scale: object.scale.to_array(),
                    selected: doc.selection.contains(&object.id),
                }
            })
            .collect();
        let materials = doc
            .scene
            .materials
            .iter()
            .map(|material| ScriptMaterial {
                name: material.name.clone(),
                diffuse_color: material.base_color,
                metallic: material.metallic,
                roughness: material.roughness,
                emissive: material.emissive,
            })
            .collect();
        Self {
            objects,
            materials,
            active,
        }
    }
}

fn primitive_script_data(primitive: &forge_core::Primitive) -> (String, Vec<f32>) {
    use forge_core::Primitive as P;
    let params = match primitive {
        P::Plane { size } | P::Cube { size } | P::Monkey { size } => vec![*size],
        P::Circle { radius, segments } => vec![*radius, *segments as f32],
        P::Grid { size, x_div, y_div } => vec![*size, *x_div as f32, *y_div as f32],
        P::IcoSphere {
            radius,
            subdivisions,
        } => vec![*radius, *subdivisions as f32],
        P::Sphere {
            radius,
            segments,
            rings,
        } => vec![*radius, *segments as f32, *rings as f32],
        P::Capsule {
            radius,
            height,
            segments,
            rings,
        } => vec![*radius, *height, *segments as f32, *rings as f32],
        P::Cylinder {
            radius,
            height,
            segments,
        }
        | P::Cone {
            radius,
            height,
            segments,
        }
        | P::Prism {
            radius,
            height,
            segments,
        }
        | P::Tube {
            radius,
            height,
            segments,
        } => vec![*radius, *height, *segments as f32],
        P::Torus {
            major_radius,
            minor_radius,
            major_segments,
            minor_segments,
        } => vec![
            *major_radius,
            *minor_radius,
            *major_segments as f32,
            *minor_segments as f32,
        ],
        P::Pyramid { size, height } => vec![*size, *height],
        P::CurveBezier { size, segments }
        | P::CurvePath {
            length: size,
            segments,
        } => vec![*size, *segments as f32],
        P::CurveCircle { radius, segments } => vec![*radius, *segments as f32],
        P::Metaball {
            radius,
            subdivisions,
        } => vec![*radius, *subdivisions as f32],
        P::Text {
            content,
            size,
            depth,
            ..
        } => vec![*size, *depth, content.len() as f32],
    };
    (primitive.name().to_owned(), params)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "python-scripting"))]
mod embedded {
    use super::{ScriptError, ScriptMaterial, ScriptObject, ScriptResult, ScriptSnapshot};
    use glam::{EulerRot, Quat};
    use pyo3::{
        exceptions::{PyIndexError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError},
        prelude::*,
        types::{PyDict, PyList, PyModule, PyTuple},
    };
    use std::{
        ffi::CString,
        path::Path,
        sync::{Arc, Mutex, OnceLock},
    };

    #[derive(Clone)]
    struct RuntimeObject {
        token: u64,
        value: ScriptObject,
    }

    #[derive(Clone, Default)]
    struct RuntimeScene {
        objects: Vec<RuntimeObject>,
        materials: Vec<ScriptMaterial>,
        active: Option<u64>,
        next_token: u64,
    }

    impl RuntimeScene {
        fn from_snapshot(snapshot: ScriptSnapshot) -> Self {
            let mut next_token = 1;
            let objects = snapshot
                .objects
                .into_iter()
                .map(|value| {
                    let token = next_token;
                    next_token += 1;
                    RuntimeObject { token, value }
                })
                .collect::<Vec<_>>();
            let active = snapshot
                .active
                .and_then(|index| objects.get(index).map(|object| object.token));
            Self {
                objects,
                materials: snapshot.materials,
                active,
                next_token,
            }
        }

        fn find(&self, token: u64) -> PyResult<&RuntimeObject> {
            self.objects
                .iter()
                .find(|object| object.token == token)
                .ok_or_else(|| PyRuntimeError::new_err("Object no longer exists"))
        }

        fn find_mut(&mut self, token: u64) -> PyResult<&mut RuntimeObject> {
            self.objects
                .iter_mut()
                .find(|object| object.token == token)
                .ok_or_else(|| PyRuntimeError::new_err("Object no longer exists"))
        }

        fn proxies(&self) -> Vec<u64> {
            self.objects.iter().map(|object| object.token).collect()
        }

        fn result(self, output: String) -> ScriptResult {
            let active = self
                .active
                .and_then(|token| self.objects.iter().position(|object| object.token == token));
            ScriptResult {
                objects: self
                    .objects
                    .into_iter()
                    .map(|object| object.value)
                    .collect(),
                materials: self.materials,
                active,
                output,
            }
        }
    }

    type SharedScene = Arc<Mutex<RuntimeScene>>;

    fn lock(scene: &SharedScene) -> PyResult<std::sync::MutexGuard<'_, RuntimeScene>> {
        scene
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Script scene state is unavailable"))
    }

    #[derive(Clone, Copy)]
    enum Transform {
        Location,
        Rotation,
        Scale,
    }

    #[pyclass(name = "Vector")]
    #[derive(Clone)]
    struct MathVector {
        values: Vec<f32>,
    }

    impl MathVector {
        fn from_values(values: Vec<f32>) -> PyResult<Self> {
            if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
                return Err(PyValueError::new_err(
                    "Vector requires finite numeric components",
                ));
            }
            Ok(Self { values })
        }

        fn compatible_values(&self, other: &Self) -> PyResult<()> {
            if self.values.len() != other.values.len() {
                return Err(PyValueError::new_err(
                    "vectors must have the same dimension",
                ));
            }
            Ok(())
        }

        fn component(&self, index: usize) -> PyResult<f32> {
            self.values
                .get(index)
                .copied()
                .ok_or_else(|| PyIndexError::new_err("vector component is unavailable"))
        }

        fn set_component(&mut self, index: usize, value: f32) -> PyResult<()> {
            if !value.is_finite() {
                return Err(PyValueError::new_err("vector components must be finite"));
            }
            let component = self
                .values
                .get_mut(index)
                .ok_or_else(|| PyIndexError::new_err("vector component is unavailable"))?;
            *component = value;
            Ok(())
        }
    }

    #[pyclass(name = "Euler")]
    #[derive(Clone)]
    struct MathEuler {
        angles: [f32; 3],
        order: String,
    }

    #[pyclass(name = "Quaternion")]
    #[derive(Clone)]
    struct MathQuaternion {
        values: [f32; 4],
    }

    fn euler_order(order: &str) -> PyResult<EulerRot> {
        match order {
            "XYZ" => Ok(EulerRot::XYZ),
            "XZY" => Ok(EulerRot::XZY),
            "YXZ" => Ok(EulerRot::YXZ),
            "YZX" => Ok(EulerRot::YZX),
            "ZXY" => Ok(EulerRot::ZXY),
            "ZYX" => Ok(EulerRot::ZYX),
            _ => Err(PyValueError::new_err(
                "Euler order must be XYZ, XZY, YXZ, YZX, ZXY, or ZYX",
            )),
        }
    }

    #[pymethods]
    impl MathEuler {
        #[new]
        #[pyo3(signature = (angles, order="XYZ"))]
        fn new(angles: Vec<f32>, order: &str) -> PyResult<Self> {
            if angles.len() != 3 || angles.iter().any(|angle| !angle.is_finite()) {
                return Err(PyValueError::new_err("Euler requires three finite angles"));
            }
            euler_order(order)?;
            Ok(Self {
                angles: [angles[0], angles[1], angles[2]],
                order: order.to_owned(),
            })
        }

        #[getter]
        fn x(&self) -> f32 {
            self.angles[0]
        }

        #[getter]
        fn y(&self) -> f32 {
            self.angles[1]
        }

        #[getter]
        fn z(&self) -> f32 {
            self.angles[2]
        }

        #[getter]
        fn order(&self) -> &str {
            &self.order
        }

        fn __len__(&self) -> usize {
            3
        }

        fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            Ok(PyTuple::new(py, self.angles)?
                .call_method0("__iter__")?
                .unbind())
        }

        fn __getitem__(&self, index: isize) -> PyResult<f32> {
            let index = if index < 0 { index + 3 } else { index };
            self.angles
                .get(usize::try_from(index).unwrap_or(usize::MAX))
                .copied()
                .ok_or_else(|| PyIndexError::new_err("Euler index out of range"))
        }

        fn to_quaternion(&self) -> PyResult<MathQuaternion> {
            let rotation = Quat::from_euler(
                euler_order(&self.order)?,
                self.angles[0],
                self.angles[1],
                self.angles[2],
            );
            Ok(MathQuaternion {
                values: [rotation.w, rotation.x, rotation.y, rotation.z],
            })
        }

        fn __repr__(&self) -> String {
            format!("Euler({:?}, '{}')", self.angles, self.order)
        }
    }

    #[pymethods]
    impl MathQuaternion {
        #[new]
        fn new(values: Vec<f32>) -> PyResult<Self> {
            if values.len() != 4 || values.iter().any(|value| !value.is_finite()) {
                return Err(PyValueError::new_err(
                    "Quaternion requires four finite components (w, x, y, z)",
                ));
            }
            Ok(Self {
                values: [values[0], values[1], values[2], values[3]],
            })
        }

        #[getter]
        fn w(&self) -> f32 {
            self.values[0]
        }

        #[getter]
        fn x(&self) -> f32 {
            self.values[1]
        }

        #[getter]
        fn y(&self) -> f32 {
            self.values[2]
        }

        #[getter]
        fn z(&self) -> f32 {
            self.values[3]
        }

        fn __len__(&self) -> usize {
            4
        }

        fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            Ok(PyTuple::new(py, self.values)?
                .call_method0("__iter__")?
                .unbind())
        }

        fn __getitem__(&self, index: isize) -> PyResult<f32> {
            let index = if index < 0 { index + 4 } else { index };
            self.values
                .get(usize::try_from(index).unwrap_or(usize::MAX))
                .copied()
                .ok_or_else(|| PyIndexError::new_err("Quaternion index out of range"))
        }

        #[pyo3(signature = (order="XYZ"))]
        fn to_euler(&self, order: &str) -> PyResult<MathEuler> {
            let rotation = Quat::from_xyzw(
                self.values[1],
                self.values[2],
                self.values[3],
                self.values[0],
            )
            .normalize();
            let (x, y, z) = rotation.to_euler(euler_order(order)?);
            Ok(MathEuler {
                angles: [x, y, z],
                order: order.to_owned(),
            })
        }

        fn __repr__(&self) -> String {
            format!("Quaternion({:?})", self.values)
        }
    }

    #[pymethods]
    impl MathVector {
        #[new]
        fn new(values: Vec<f32>) -> PyResult<Self> {
            Self::from_values(values)
        }

        #[getter]
        fn x(&self) -> PyResult<f32> {
            self.component(0)
        }

        #[setter]
        fn set_x(&mut self, value: f32) -> PyResult<()> {
            self.set_component(0, value)
        }

        #[getter]
        fn y(&self) -> PyResult<f32> {
            self.component(1)
        }

        #[setter]
        fn set_y(&mut self, value: f32) -> PyResult<()> {
            self.set_component(1, value)
        }

        #[getter]
        fn z(&self) -> PyResult<f32> {
            self.component(2)
        }

        #[setter]
        fn set_z(&mut self, value: f32) -> PyResult<()> {
            self.set_component(2, value)
        }

        #[getter]
        fn w(&self) -> PyResult<f32> {
            self.component(3)
        }

        #[setter]
        fn set_w(&mut self, value: f32) -> PyResult<()> {
            self.set_component(3, value)
        }

        #[getter]
        fn length(&self) -> f32 {
            self.length_squared().sqrt()
        }

        #[getter]
        fn length_squared(&self) -> f32 {
            self.values.iter().map(|value| value * value).sum()
        }

        fn __len__(&self) -> usize {
            self.values.len()
        }

        fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            Ok(PyTuple::new(py, &self.values)?
                .call_method0("__iter__")?
                .unbind())
        }

        fn __getitem__(&self, index: isize) -> PyResult<f32> {
            let index = if index < 0 {
                index + self.values.len() as isize
            } else {
                index
            };
            self.values
                .get(usize::try_from(index).unwrap_or(usize::MAX))
                .copied()
                .ok_or_else(|| PyIndexError::new_err("vector index out of range"))
        }

        fn __setitem__(&mut self, index: isize, value: f32) -> PyResult<()> {
            let index = if index < 0 {
                index + self.values.len() as isize
            } else {
                index
            };
            let index = usize::try_from(index)
                .map_err(|_| PyIndexError::new_err("vector index out of range"))?;
            self.set_component(index, value)
        }

        fn __add__(&self, other: PyRef<'_, MathVector>) -> PyResult<Self> {
            self.compatible_values(&other)?;
            Self::from_values(
                self.values
                    .iter()
                    .zip(other.values.iter())
                    .map(|(left, right)| left + right)
                    .collect(),
            )
        }

        fn __sub__(&self, other: PyRef<'_, MathVector>) -> PyResult<Self> {
            self.compatible_values(&other)?;
            Self::from_values(
                self.values
                    .iter()
                    .zip(other.values.iter())
                    .map(|(left, right)| left - right)
                    .collect(),
            )
        }

        fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
            if let Ok(scale) = other.extract::<f32>() {
                return Self::from_values(self.values.iter().map(|value| value * scale).collect());
            }
            if let Ok(vector) = other.extract::<PyRef<'_, MathVector>>() {
                self.compatible_values(&vector)?;
                return Self::from_values(
                    self.values
                        .iter()
                        .zip(vector.values.iter())
                        .map(|(left, right)| left * right)
                        .collect(),
                );
            }
            Err(PyTypeError::new_err(
                "Vector can be multiplied by a number or Vector",
            ))
        }

        fn __rmul__(&self, other: f32) -> PyResult<Self> {
            Self::from_values(self.values.iter().map(|value| value * other).collect())
        }

        fn __truediv__(&self, divisor: f32) -> PyResult<Self> {
            if divisor == 0.0 || !divisor.is_finite() {
                return Err(PyValueError::new_err("invalid Vector divisor"));
            }
            Self::from_values(self.values.iter().map(|value| value / divisor).collect())
        }

        fn __neg__(&self) -> PyResult<Self> {
            Self::from_values(self.values.iter().map(|value| -value).collect())
        }

        fn __eq__(&self, other: PyRef<'_, MathVector>) -> bool {
            self.values == other.values
        }

        fn dot(&self, other: PyRef<'_, MathVector>) -> PyResult<f32> {
            self.compatible_values(&other)?;
            Ok(self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(left, right)| left * right)
                .sum())
        }

        fn cross(&self, other: PyRef<'_, MathVector>) -> PyResult<Self> {
            self.compatible_values(&other)?;
            if self.values.len() != 3 {
                return Err(PyValueError::new_err("cross product requires 3D vectors"));
            }
            let a = glam::Vec3::from_slice(&self.values);
            let b = glam::Vec3::from_slice(&other.values);
            Self::from_values(a.cross(b).to_array().to_vec())
        }

        #[pyo3(signature = (other, fallback=None))]
        fn angle(&self, other: PyRef<'_, MathVector>, fallback: Option<f32>) -> PyResult<f32> {
            self.compatible_values(&other)?;
            let left_length = self.length();
            let right_length = other.length();
            if left_length == 0.0 || right_length == 0.0 {
                return fallback.ok_or_else(|| {
                    PyValueError::new_err("angle is undefined for a zero-length Vector")
                });
            }
            let cosine = self.dot(other)? / (left_length * right_length);
            Ok(cosine.clamp(-1.0, 1.0).acos())
        }

        fn lerp(&self, other: PyRef<'_, MathVector>, factor: f32) -> PyResult<Self> {
            self.compatible_values(&other)?;
            Self::from_values(
                self.values
                    .iter()
                    .zip(other.values.iter())
                    .map(|(left, right)| left + (right - left) * factor)
                    .collect(),
            )
        }

        fn project(&self, other: PyRef<'_, MathVector>) -> PyResult<Self> {
            self.compatible_values(&other)?;
            let denominator = other.length_squared();
            if denominator == 0.0 {
                return Err(PyValueError::new_err(
                    "cannot project onto a zero-length Vector",
                ));
            }
            let dot = self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(left, right)| left * right)
                .sum::<f32>();
            let scale = dot / denominator;
            Self::from_values(other.values.iter().map(|value| value * scale).collect())
        }

        fn reflect(&self, normal: PyRef<'_, MathVector>) -> PyResult<Self> {
            self.compatible_values(&normal)?;
            let normal_length = normal.length();
            if normal_length == 0.0 {
                return Err(PyValueError::new_err(
                    "cannot reflect around a zero-length normal",
                ));
            }
            let dot = self
                .values
                .iter()
                .zip(normal.values.iter())
                .map(|(left, right)| left * right)
                .sum::<f32>();
            let scale = 2.0 * dot / (normal_length * normal_length);
            Self::from_values(
                self.values
                    .iter()
                    .zip(normal.values.iter())
                    .map(|(value, component)| value - scale * component)
                    .collect(),
            )
        }

        fn normalize(&mut self) -> PyResult<()> {
            let length = self
                .values
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            if length == 0.0 {
                return Err(PyValueError::new_err(
                    "cannot normalize a zero-length Vector",
                ));
            }
            for value in &mut self.values {
                *value /= length;
            }
            Ok(())
        }

        fn normalized(&self) -> PyResult<Self> {
            let mut result = self.clone();
            result.normalize()?;
            Ok(result)
        }

        fn copy(&self) -> Self {
            self.clone()
        }

        fn __repr__(&self) -> String {
            format!("Vector({:?})", self.values)
        }
    }

    #[pyclass]
    struct VectorProxy {
        scene: SharedScene,
        token: u64,
        transform: Transform,
    }

    impl VectorProxy {
        fn values(&self) -> PyResult<[f32; 3]> {
            let scene = lock(&self.scene)?;
            let object = scene.find(self.token)?;
            Ok(match self.transform {
                Transform::Location => object.value.location,
                Transform::Rotation => object.value.rotation_euler,
                Transform::Scale => object.value.scale,
            })
        }

        fn set_component(&self, index: usize, value: f32) -> PyResult<()> {
            if index >= 3 {
                return Err(PyIndexError::new_err("vector index out of range"));
            }
            let mut scene = lock(&self.scene)?;
            let object = scene.find_mut(self.token)?;
            let target = match self.transform {
                Transform::Location => &mut object.value.location,
                Transform::Rotation => &mut object.value.rotation_euler,
                Transform::Scale => &mut object.value.scale,
            };
            target[index] = value;
            Ok(())
        }
    }

    #[pymethods]
    impl VectorProxy {
        #[getter]
        fn x(&self) -> PyResult<f32> {
            Ok(self.values()?[0])
        }

        #[setter]
        fn set_x(&self, value: f32) -> PyResult<()> {
            self.set_component(0, value)
        }

        #[getter]
        fn y(&self) -> PyResult<f32> {
            Ok(self.values()?[1])
        }

        #[setter]
        fn set_y(&self, value: f32) -> PyResult<()> {
            self.set_component(1, value)
        }

        #[getter]
        fn z(&self) -> PyResult<f32> {
            Ok(self.values()?[2])
        }

        #[setter]
        fn set_z(&self, value: f32) -> PyResult<()> {
            self.set_component(2, value)
        }

        fn __len__(&self) -> usize {
            3
        }

        fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            let values = self.values()?;
            Ok(PyTuple::new(py, values)?.call_method0("__iter__")?.unbind())
        }

        fn __getitem__(&self, index: isize) -> PyResult<f32> {
            let index = if index < 0 { index + 3 } else { index };
            if !(0..3).contains(&index) {
                return Err(PyIndexError::new_err("vector index out of range"));
            }
            Ok(self.values()?[index as usize])
        }

        fn __setitem__(&self, index: isize, value: f32) -> PyResult<()> {
            let index = if index < 0 { index + 3 } else { index };
            if !(0..3).contains(&index) {
                return Err(PyIndexError::new_err("vector index out of range"));
            }
            self.set_component(index as usize, value)
        }
    }

    #[pyclass]
    struct SceneObjectProxy {
        scene: SharedScene,
        token: u64,
    }

    impl SceneObjectProxy {
        fn vector(&self, py: Python<'_>, transform: Transform) -> PyResult<Py<VectorProxy>> {
            Py::new(
                py,
                VectorProxy {
                    scene: self.scene.clone(),
                    token: self.token,
                    transform,
                },
            )
        }

        fn set_vector(&self, transform: Transform, values: Vec<f32>) -> PyResult<()> {
            if values.len() != 3 {
                return Err(PyValueError::new_err("expected a 3-component vector"));
            }
            let values = [values[0], values[1], values[2]];
            let mut scene = lock(&self.scene)?;
            let object = scene.find_mut(self.token)?;
            match transform {
                Transform::Location => object.value.location = values,
                Transform::Rotation => object.value.rotation_euler = values,
                Transform::Scale => object.value.scale = values,
            }
            Ok(())
        }
    }

    #[pymethods]
    impl SceneObjectProxy {
        #[getter]
        fn name(&self) -> PyResult<String> {
            Ok(lock(&self.scene)?.find(self.token)?.value.name.clone())
        }

        #[setter]
        fn set_name(&self, value: String) -> PyResult<()> {
            lock(&self.scene)?.find_mut(self.token)?.value.name = value;
            Ok(())
        }

        #[getter]
        fn r#type(&self) -> PyResult<String> {
            Ok(lock(&self.scene)?
                .find(self.token)?
                .value
                .object_type
                .clone())
        }

        #[getter]
        fn location(&self, py: Python<'_>) -> PyResult<Py<VectorProxy>> {
            self.vector(py, Transform::Location)
        }

        #[setter]
        fn set_location(&self, values: Vec<f32>) -> PyResult<()> {
            self.set_vector(Transform::Location, values)
        }

        #[getter]
        fn rotation_euler(&self, py: Python<'_>) -> PyResult<Py<VectorProxy>> {
            self.vector(py, Transform::Rotation)
        }

        #[setter]
        fn set_rotation_euler(&self, values: Vec<f32>) -> PyResult<()> {
            self.set_vector(Transform::Rotation, values)
        }

        #[getter]
        fn rotation_quaternion(&self, py: Python<'_>) -> PyResult<Py<MathQuaternion>> {
            let euler = lock(&self.scene)?.find(self.token)?.value.rotation_euler;
            let rotation = Quat::from_euler(EulerRot::XYZ, euler[0], euler[1], euler[2]);
            Py::new(
                py,
                MathQuaternion {
                    values: [rotation.w, rotation.x, rotation.y, rotation.z],
                },
            )
        }

        #[setter]
        fn set_rotation_quaternion(&self, values: Vec<f32>) -> PyResult<()> {
            if values.len() != 4 || values.iter().any(|value| !value.is_finite()) {
                return Err(PyValueError::new_err(
                    "rotation_quaternion requires four finite components",
                ));
            }
            let rotation = Quat::from_xyzw(values[1], values[2], values[3], values[0]);
            if rotation.length_squared() == 0.0 {
                return Err(PyValueError::new_err("rotation_quaternion cannot be zero"));
            }
            let (x, y, z) = rotation.normalize().to_euler(EulerRot::XYZ);
            lock(&self.scene)?
                .find_mut(self.token)?
                .value
                .rotation_euler = [x, y, z];
            Ok(())
        }

        #[getter]
        fn scale(&self, py: Python<'_>) -> PyResult<Py<VectorProxy>> {
            self.vector(py, Transform::Scale)
        }

        #[setter]
        fn set_scale(&self, values: Vec<f32>) -> PyResult<()> {
            self.set_vector(Transform::Scale, values)
        }

        fn select_set(&self, state: bool) -> PyResult<()> {
            lock(&self.scene)?.find_mut(self.token)?.value.selected = state;
            Ok(())
        }

        fn select_get(&self) -> PyResult<bool> {
            Ok(lock(&self.scene)?.find(self.token)?.value.selected)
        }

        #[getter]
        fn data(&self, py: Python<'_>) -> PyResult<Py<ObjectDataProxy>> {
            Py::new(
                py,
                ObjectDataProxy {
                    scene: self.scene.clone(),
                    token: self.token,
                },
            )
        }

        #[getter]
        fn active_material(&self, py: Python<'_>) -> PyResult<Option<Py<MaterialProxy>>> {
            let index = lock(&self.scene)?.find(self.token)?.value.material;
            index
                .map(|index| {
                    Py::new(
                        py,
                        MaterialProxy {
                            scene: self.scene.clone(),
                            index,
                        },
                    )
                })
                .transpose()
        }

        #[setter]
        fn set_active_material(
            &self,
            py: Python<'_>,
            material: Option<Py<MaterialProxy>>,
        ) -> PyResult<()> {
            let index = material.map(|material| {
                let material = material.borrow(py);
                if !Arc::ptr_eq(&self.scene, &material.scene) {
                    return Err(PyValueError::new_err(
                        "material belongs to a different script scene",
                    ));
                }
                Ok(material.index)
            });
            let index = index.transpose()?;
            let mut scene = lock(&self.scene)?;
            if index.is_some_and(|index| index >= scene.materials.len()) {
                return Err(PyIndexError::new_err("material index out of range"));
            }
            scene.find_mut(self.token)?.value.material = index;
            Ok(())
        }

        fn __repr__(&self) -> PyResult<String> {
            Ok(format!("<Object '{}' >", self.name()?))
        }
    }

    #[pyclass]
    struct ObjectCollection {
        scene: SharedScene,
    }

    impl ObjectCollection {
        fn proxy(&self, py: Python<'_>, token: u64) -> PyResult<Py<SceneObjectProxy>> {
            Py::new(
                py,
                SceneObjectProxy {
                    scene: self.scene.clone(),
                    token,
                },
            )
        }

        fn tokens(&self) -> PyResult<Vec<u64>> {
            Ok(lock(&self.scene)?.proxies())
        }
    }

    #[pymethods]
    impl ObjectCollection {
        #[pyo3(signature = (name, default=None))]
        fn get(
            &self,
            py: Python<'_>,
            name: &str,
            default: Option<Py<PyAny>>,
        ) -> PyResult<Py<PyAny>> {
            let token = lock(&self.scene)?
                .objects
                .iter()
                .find(|object| object.value.name == name)
                .map(|object| object.token);
            match token {
                Some(token) => Ok(self.proxy(py, token)?.into_any()),
                None => Ok(default.unwrap_or_else(|| py.None())),
            }
        }

        fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            let objects = self
                .tokens()?
                .into_iter()
                .map(|token| self.proxy(py, token))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, objects)?.call_method0("__iter__")?.unbind())
        }

        fn __len__(&self) -> PyResult<usize> {
            Ok(lock(&self.scene)?.objects.len())
        }

        fn __getitem__(
            &self,
            py: Python<'_>,
            key: &Bound<'_, PyAny>,
        ) -> PyResult<Py<SceneObjectProxy>> {
            if let Ok(name) = key.extract::<String>() {
                let token = lock(&self.scene)?
                    .objects
                    .iter()
                    .find(|object| object.value.name == name)
                    .map(|object| object.token)
                    .ok_or_else(|| PyKeyError::new_err(name))?;
                return self.proxy(py, token);
            }
            if let Ok(index) = key.extract::<isize>() {
                let tokens = self.tokens()?;
                let index = if index < 0 {
                    index + tokens.len() as isize
                } else {
                    index
                };
                let token = tokens
                    .get(usize::try_from(index).unwrap_or(usize::MAX))
                    .copied()
                    .ok_or_else(|| PyIndexError::new_err("object index out of range"))?;
                return self.proxy(py, token);
            }
            Err(PyTypeError::new_err(
                "object lookup requires a name or index",
            ))
        }

        fn __contains__(&self, name: &str) -> PyResult<bool> {
            Ok(lock(&self.scene)?
                .objects
                .iter()
                .any(|object| object.value.name == name))
        }
    }

    #[pyclass]
    struct ContextProxy {
        scene: SharedScene,
    }

    impl ContextProxy {
        fn active_proxy(&self, py: Python<'_>) -> PyResult<Option<Py<SceneObjectProxy>>> {
            let token = lock(&self.scene)?.active;
            token
                .map(|token| {
                    Py::new(
                        py,
                        SceneObjectProxy {
                            scene: self.scene.clone(),
                            token,
                        },
                    )
                })
                .transpose()
        }
    }

    #[pyclass]
    struct ViewLayerProxy {
        scene: SharedScene,
    }

    #[pyclass]
    struct SceneProxy {
        scene: SharedScene,
    }

    #[pymethods]
    impl SceneProxy {
        #[getter]
        fn objects(&self, py: Python<'_>) -> PyResult<Py<ObjectCollection>> {
            Py::new(
                py,
                ObjectCollection {
                    scene: self.scene.clone(),
                },
            )
        }
    }

    #[pymethods]
    impl ViewLayerProxy {
        #[getter]
        fn objects(&self, py: Python<'_>) -> PyResult<Py<ViewLayerObjectsProxy>> {
            Py::new(
                py,
                ViewLayerObjectsProxy {
                    scene: self.scene.clone(),
                },
            )
        }
    }

    #[pyclass]
    struct ViewLayerObjectsProxy {
        scene: SharedScene,
    }

    #[pymethods]
    impl ViewLayerObjectsProxy {
        #[getter]
        fn active(&self, py: Python<'_>) -> PyResult<Option<Py<SceneObjectProxy>>> {
            let token = lock(&self.scene)?.active;
            token
                .map(|token| {
                    Py::new(
                        py,
                        SceneObjectProxy {
                            scene: self.scene.clone(),
                            token,
                        },
                    )
                })
                .transpose()
        }

        #[setter]
        fn set_active(&self, py: Python<'_>, value: Option<Py<SceneObjectProxy>>) -> PyResult<()> {
            let token = value.map(|object| object.borrow(py).token);
            lock(&self.scene)?.active = token;
            Ok(())
        }
    }

    #[pymethods]
    impl ContextProxy {
        #[getter]
        fn object(&self, py: Python<'_>) -> PyResult<Option<Py<SceneObjectProxy>>> {
            self.active_proxy(py)
        }

        #[getter]
        fn active_object(&self, py: Python<'_>) -> PyResult<Option<Py<SceneObjectProxy>>> {
            self.active_proxy(py)
        }

        #[getter]
        fn selected_objects(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
            let tokens = {
                let scene = lock(&self.scene)?;
                scene
                    .objects
                    .iter()
                    .filter(|object| object.value.selected)
                    .map(|object| object.token)
                    .collect::<Vec<_>>()
            };
            let objects = tokens
                .into_iter()
                .map(|token| {
                    Py::new(
                        py,
                        SceneObjectProxy {
                            scene: self.scene.clone(),
                            token,
                        },
                    )
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, objects)?.unbind())
        }

        #[getter]
        fn view_layer(&self, py: Python<'_>) -> PyResult<Py<ViewLayerProxy>> {
            Py::new(
                py,
                ViewLayerProxy {
                    scene: self.scene.clone(),
                },
            )
        }

        #[getter]
        fn scene(&self, py: Python<'_>) -> PyResult<Py<SceneProxy>> {
            Py::new(
                py,
                SceneProxy {
                    scene: self.scene.clone(),
                },
            )
        }
    }

    #[pyclass]
    struct DataProxy {
        scene: SharedScene,
    }

    #[pyclass]
    struct MaterialCollection {
        scene: SharedScene,
    }

    impl MaterialCollection {
        fn proxy(&self, py: Python<'_>, index: usize) -> PyResult<Py<MaterialProxy>> {
            if index >= lock(&self.scene)?.materials.len() {
                return Err(PyIndexError::new_err("material index out of range"));
            }
            Py::new(
                py,
                MaterialProxy {
                    scene: self.scene.clone(),
                    index,
                },
            )
        }
    }

    #[pymethods]
    impl MaterialCollection {
        #[pyo3(signature = (name, default=None))]
        fn get(
            &self,
            py: Python<'_>,
            name: &str,
            default: Option<Py<PyAny>>,
        ) -> PyResult<Py<PyAny>> {
            let index = lock(&self.scene)?
                .materials
                .iter()
                .position(|material| material.name == name);
            match index {
                Some(index) => Ok(self.proxy(py, index)?.into_any()),
                None => Ok(default.unwrap_or_else(|| py.None())),
            }
        }

        #[pyo3(signature = (name="Material"))]
        fn new(&self, py: Python<'_>, name: &str) -> PyResult<Py<MaterialProxy>> {
            let mut scene = lock(&self.scene)?;
            let name = unique_material_name(&name, &scene.materials);
            scene.materials.push(ScriptMaterial {
                name,
                diffuse_color: [0.8, 0.8, 0.8, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                emissive: [0.0; 3],
            });
            let index = scene.materials.len() - 1;
            drop(scene);
            self.proxy(py, index)
        }

        fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            let count = lock(&self.scene)?.materials.len();
            let materials = (0..count)
                .map(|index| self.proxy(py, index))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, materials)?
                .call_method0("__iter__")?
                .unbind())
        }

        fn __len__(&self) -> PyResult<usize> {
            Ok(lock(&self.scene)?.materials.len())
        }

        fn __getitem__(
            &self,
            py: Python<'_>,
            key: &Bound<'_, PyAny>,
        ) -> PyResult<Py<MaterialProxy>> {
            if let Ok(name) = key.extract::<String>() {
                let index = lock(&self.scene)?
                    .materials
                    .iter()
                    .position(|material| material.name == name)
                    .ok_or_else(|| PyKeyError::new_err(name))?;
                return self.proxy(py, index);
            }
            if let Ok(index) = key.extract::<isize>() {
                let count = lock(&self.scene)?.materials.len();
                let index = if index < 0 {
                    index + count as isize
                } else {
                    index
                };
                return self.proxy(
                    py,
                    usize::try_from(index)
                        .map_err(|_| PyIndexError::new_err("material index out of range"))?,
                );
            }
            Err(PyTypeError::new_err(
                "material lookup requires a name or index",
            ))
        }
    }

    fn unique_material_name(name: &str, materials: &[ScriptMaterial]) -> String {
        if !materials.iter().any(|material| material.name == name) {
            return name.to_owned();
        }
        let mut suffix = 1;
        loop {
            let candidate = format!("{name}.{suffix:03}");
            if !materials.iter().any(|material| material.name == candidate) {
                return candidate;
            }
            suffix += 1;
        }
    }

    #[pyclass]
    struct MaterialProxy {
        scene: SharedScene,
        index: usize,
    }

    #[pymethods]
    impl MaterialProxy {
        #[getter]
        fn name(&self) -> PyResult<String> {
            Ok(lock(&self.scene)?
                .materials
                .get(self.index)
                .ok_or_else(|| PyIndexError::new_err("material no longer exists"))?
                .name
                .clone())
        }

        #[setter]
        fn set_name(&self, value: String) -> PyResult<()> {
            lock(&self.scene)?
                .materials
                .get_mut(self.index)
                .ok_or_else(|| PyIndexError::new_err("material no longer exists"))?
                .name = value;
            Ok(())
        }

        #[getter]
        fn diffuse_color(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            let color = lock(&self.scene)?
                .materials
                .get(self.index)
                .ok_or_else(|| PyIndexError::new_err("material no longer exists"))?
                .diffuse_color;
            Ok(PyTuple::new(py, color)?.unbind())
        }

        #[setter]
        fn set_diffuse_color(&self, values: Vec<f32>) -> PyResult<()> {
            if values.len() != 4 || values.iter().any(|value| !value.is_finite()) {
                return Err(PyValueError::new_err(
                    "diffuse_color requires four finite components",
                ));
            }
            let mut scene = lock(&self.scene)?;
            let material = scene
                .materials
                .get_mut(self.index)
                .ok_or_else(|| PyIndexError::new_err("material no longer exists"))?;
            material.diffuse_color = [values[0], values[1], values[2], values[3]];
            Ok(())
        }

        #[getter]
        fn metallic(&self) -> PyResult<f32> {
            Ok(lock(&self.scene)?
                .materials
                .get(self.index)
                .ok_or_else(|| PyIndexError::new_err("material no longer exists"))?
                .metallic)
        }

        #[setter]
        fn set_metallic(&self, value: f32) -> PyResult<()> {
            if !value.is_finite() {
                return Err(PyValueError::new_err("metallic must be finite"));
            }
            lock(&self.scene)?
                .materials
                .get_mut(self.index)
                .ok_or_else(|| PyIndexError::new_err("material no longer exists"))?
                .metallic = value.clamp(0.0, 1.0);
            Ok(())
        }

        #[getter]
        fn roughness(&self) -> PyResult<f32> {
            Ok(lock(&self.scene)?
                .materials
                .get(self.index)
                .ok_or_else(|| PyIndexError::new_err("material no longer exists"))?
                .roughness)
        }

        #[setter]
        fn set_roughness(&self, value: f32) -> PyResult<()> {
            if !value.is_finite() {
                return Err(PyValueError::new_err("roughness must be finite"));
            }
            lock(&self.scene)?
                .materials
                .get_mut(self.index)
                .ok_or_else(|| PyIndexError::new_err("material no longer exists"))?
                .roughness = value.clamp(0.0, 1.0);
            Ok(())
        }
    }

    #[pyclass]
    struct ObjectDataProxy {
        scene: SharedScene,
        token: u64,
    }

    #[pymethods]
    impl ObjectDataProxy {
        #[getter]
        fn materials(&self, py: Python<'_>) -> PyResult<Py<ObjectMaterialSlots>> {
            Py::new(
                py,
                ObjectMaterialSlots {
                    scene: self.scene.clone(),
                    token: self.token,
                },
            )
        }
    }

    #[pyclass]
    struct ObjectMaterialSlots {
        scene: SharedScene,
        token: u64,
    }

    #[pymethods]
    impl ObjectMaterialSlots {
        fn append(&self, py: Python<'_>, material: Py<MaterialProxy>) -> PyResult<()> {
            let material = material.borrow(py);
            if !Arc::ptr_eq(&self.scene, &material.scene) {
                return Err(PyValueError::new_err(
                    "material belongs to a different script scene",
                ));
            }
            let mut scene = lock(&self.scene)?;
            if material.index >= scene.materials.len() {
                return Err(PyIndexError::new_err("material index out of range"));
            }
            let object = scene.find_mut(self.token)?;
            if object.value.material.is_some() {
                return Err(PyValueError::new_err(
                    "BlendCraft currently supports one material slot per object",
                ));
            }
            object.value.material = Some(material.index);
            Ok(())
        }

        fn clear(&self) -> PyResult<()> {
            lock(&self.scene)?.find_mut(self.token)?.value.material = None;
            Ok(())
        }

        fn __len__(&self) -> PyResult<usize> {
            Ok(usize::from(
                lock(&self.scene)?
                    .find(self.token)?
                    .value
                    .material
                    .is_some(),
            ))
        }

        fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<MaterialProxy>> {
            if index != 0 && index != -1 {
                return Err(PyIndexError::new_err("material slot index out of range"));
            }
            let material_index = lock(&self.scene)?
                .find(self.token)?
                .value
                .material
                .ok_or_else(|| PyIndexError::new_err("object has no material slot"))?;
            Py::new(
                py,
                MaterialProxy {
                    scene: self.scene.clone(),
                    index: material_index,
                },
            )
        }
    }

    #[pymethods]
    impl DataProxy {
        #[getter]
        fn objects(&self, py: Python<'_>) -> PyResult<Py<ObjectCollection>> {
            Py::new(
                py,
                ObjectCollection {
                    scene: self.scene.clone(),
                },
            )
        }

        #[getter]
        fn materials(&self, py: Python<'_>) -> PyResult<Py<MaterialCollection>> {
            Py::new(
                py,
                MaterialCollection {
                    scene: self.scene.clone(),
                },
            )
        }
    }

    #[pyclass]
    struct ObjectOps {
        scene: SharedScene,
    }

    #[pymethods]
    impl ObjectOps {
        #[pyo3(signature = (action="DESELECT"))]
        fn select_all(&self, action: &str) -> PyResult<&'static str> {
            let mut scene = lock(&self.scene)?;
            match action {
                "DESELECT" => {
                    for object in &mut scene.objects {
                        object.value.selected = false;
                    }
                }
                "SELECT" => {
                    for object in &mut scene.objects {
                        object.value.selected = true;
                    }
                }
                "INVERT" => {
                    for object in &mut scene.objects {
                        object.value.selected = !object.value.selected;
                    }
                }
                _ => return Err(PyValueError::new_err("unsupported select_all action")),
            }
            Ok("FINISHED")
        }

        fn delete(&self) -> PyResult<&'static str> {
            let mut scene = lock(&self.scene)?;
            scene.objects.retain(|object| !object.value.selected);
            if scene
                .active
                .is_some_and(|active| !scene.objects.iter().any(|object| object.token == active))
            {
                scene.active = None;
            }
            Ok("FINISHED")
        }
    }

    #[pyclass]
    struct MeshOps {
        scene: SharedScene,
    }

    impl MeshOps {
        fn add(
            &self,
            primitive: &str,
            default_name: &str,
            params: Vec<f32>,
            location: Option<Vec<f32>>,
            rotation: Option<Vec<f32>>,
        ) -> PyResult<&'static str> {
            let location = vector_or_default(location, [0.0; 3])?;
            let rotation = vector_or_default(rotation, [0.0; 3])?;
            let mut scene = lock(&self.scene)?;
            let token = scene.next_token;
            scene.next_token += 1;
            let name = unique_object_name(default_name, &scene.objects);
            scene.objects.push(RuntimeObject {
                token,
                value: ScriptObject {
                    id: None,
                    name,
                    object_type: "MESH".to_owned(),
                    primitive: primitive.to_owned(),
                    primitive_params: params,
                    material: None,
                    location,
                    rotation_euler: rotation,
                    scale: [1.0; 3],
                    selected: true,
                },
            });
            for object in &mut scene.objects {
                object.value.selected = object.token == token;
            }
            scene.active = Some(token);
            Ok("FINISHED")
        }
    }

    #[pymethods]
    impl MeshOps {
        #[pyo3(signature = (size=2.0, location=None, rotation=None))]
        fn primitive_cube_add(
            &self,
            size: f32,
            location: Option<Vec<f32>>,
            rotation: Option<Vec<f32>>,
        ) -> PyResult<&'static str> {
            positive(size, "size")?;
            self.add("Cube", "Cube", vec![size], location, rotation)
        }

        #[pyo3(signature = (size=2.0, location=None, rotation=None))]
        fn primitive_plane_add(
            &self,
            size: f32,
            location: Option<Vec<f32>>,
            rotation: Option<Vec<f32>>,
        ) -> PyResult<&'static str> {
            positive(size, "size")?;
            self.add("Plane", "Plane", vec![size], location, rotation)
        }

        #[pyo3(signature = (segments=32, ring_count=16, radius=1.0, location=None, rotation=None))]
        fn primitive_uv_sphere_add(
            &self,
            segments: u32,
            ring_count: u32,
            radius: f32,
            location: Option<Vec<f32>>,
            rotation: Option<Vec<f32>>,
        ) -> PyResult<&'static str> {
            positive(radius, "radius")?;
            valid_segments(segments)?;
            valid_segments(ring_count)?;
            self.add(
                "Sphere",
                "Sphere",
                vec![radius, segments as f32, ring_count as f32],
                location,
                rotation,
            )
        }

        #[pyo3(signature = (subdivisions=2, radius=1.0, location=None, rotation=None))]
        fn primitive_ico_sphere_add(
            &self,
            subdivisions: u32,
            radius: f32,
            location: Option<Vec<f32>>,
            rotation: Option<Vec<f32>>,
        ) -> PyResult<&'static str> {
            positive(radius, "radius")?;
            if subdivisions > 8 {
                return Err(PyValueError::new_err(
                    "icosphere subdivisions are limited to 8",
                ));
            }
            self.add(
                "Ico Sphere",
                "Icosphere",
                vec![radius, subdivisions as f32],
                location,
                rotation,
            )
        }

        #[pyo3(signature = (vertices=32, radius=1.0, depth=2.0, location=None, rotation=None))]
        fn primitive_cylinder_add(
            &self,
            vertices: u32,
            radius: f32,
            depth: f32,
            location: Option<Vec<f32>>,
            rotation: Option<Vec<f32>>,
        ) -> PyResult<&'static str> {
            valid_segments(vertices)?;
            positive(radius, "radius")?;
            positive(depth, "depth")?;
            self.add(
                "Cylinder",
                "Cylinder",
                vec![radius, depth, vertices as f32],
                location,
                rotation,
            )
        }

        #[pyo3(signature = (vertices=32, radius1=1.0, radius2=0.0, depth=2.0, location=None, rotation=None))]
        fn primitive_cone_add(
            &self,
            vertices: u32,
            radius1: f32,
            radius2: f32,
            depth: f32,
            location: Option<Vec<f32>>,
            rotation: Option<Vec<f32>>,
        ) -> PyResult<&'static str> {
            valid_segments(vertices)?;
            positive(radius1, "radius1")?;
            positive(depth, "depth")?;
            if radius2 != 0.0 {
                return Err(PyValueError::new_err(
                    "truncated cones (nonzero radius2) are not supported yet",
                ));
            }
            self.add(
                "Cone",
                "Cone",
                vec![radius1, depth, vertices as f32],
                location,
                rotation,
            )
        }

        #[pyo3(signature = (major_segments=48, minor_segments=12, major_radius=1.0, minor_radius=0.25, location=None, rotation=None))]
        fn primitive_torus_add(
            &self,
            major_segments: u32,
            minor_segments: u32,
            major_radius: f32,
            minor_radius: f32,
            location: Option<Vec<f32>>,
            rotation: Option<Vec<f32>>,
        ) -> PyResult<&'static str> {
            valid_segments(major_segments)?;
            valid_segments(minor_segments)?;
            positive(major_radius, "major_radius")?;
            positive(minor_radius, "minor_radius")?;
            self.add(
                "Torus",
                "Torus",
                vec![
                    major_radius,
                    minor_radius,
                    major_segments as f32,
                    minor_segments as f32,
                ],
                location,
                rotation,
            )
        }

        #[pyo3(signature = (size=2.0, location=None, rotation=None))]
        fn primitive_monkey_add(
            &self,
            size: f32,
            location: Option<Vec<f32>>,
            rotation: Option<Vec<f32>>,
        ) -> PyResult<&'static str> {
            positive(size, "size")?;
            self.add("Monkey", "Suzanne", vec![size], location, rotation)
        }
    }

    fn vector_or_default(value: Option<Vec<f32>>, default: [f32; 3]) -> PyResult<[f32; 3]> {
        match value {
            Some(value) if value.len() == 3 => Ok([value[0], value[1], value[2]]),
            Some(_) => Err(PyValueError::new_err("expected a 3-component vector")),
            None => Ok(default),
        }
    }

    fn positive(value: f32, name: &str) -> PyResult<()> {
        if !value.is_finite() || value <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "{name} must be a positive finite number"
            )));
        }
        Ok(())
    }

    fn valid_segments(value: u32) -> PyResult<()> {
        if !(3..=512).contains(&value) {
            return Err(PyValueError::new_err(
                "mesh segment counts must be between 3 and 512",
            ));
        }
        Ok(())
    }

    fn unique_object_name(base_name: &str, objects: &[RuntimeObject]) -> String {
        if !objects.iter().any(|object| object.value.name == base_name) {
            return base_name.to_owned();
        }
        let mut suffix = 1;
        loop {
            let name = format!("{base_name}.{suffix:03}");
            if !objects.iter().any(|object| object.value.name == name) {
                return name;
            }
            suffix += 1;
        }
    }

    #[pyclass]
    struct OpsProxy {
        scene: SharedScene,
    }

    #[pymethods]
    impl OpsProxy {
        #[getter]
        fn object(&self, py: Python<'_>) -> PyResult<Py<ObjectOps>> {
            Py::new(
                py,
                ObjectOps {
                    scene: self.scene.clone(),
                },
            )
        }

        #[getter]
        fn mesh(&self, py: Python<'_>) -> PyResult<Py<MeshOps>> {
            Py::new(
                py,
                MeshOps {
                    scene: self.scene.clone(),
                },
            )
        }
    }

    pub fn execute(source: String, snapshot: ScriptSnapshot) -> Result<ScriptResult, ScriptError> {
        configure_bundled_python()?;
        let scene = Arc::new(Mutex::new(RuntimeScene::from_snapshot(snapshot)));
        Python::with_gil(|py| {
            let module = PyModule::new(py, "bpy").map_err(|error| ScriptError {
                output: String::new(),
                message: error.to_string(),
            })?;
            let data = Py::new(
                py,
                DataProxy {
                    scene: scene.clone(),
                },
            )
            .map_err(|error| ScriptError {
                output: String::new(),
                message: error.to_string(),
            })?;
            let context = Py::new(
                py,
                ContextProxy {
                    scene: scene.clone(),
                },
            )
            .map_err(|error| ScriptError {
                output: String::new(),
                message: error.to_string(),
            })?;
            let ops = Py::new(
                py,
                OpsProxy {
                    scene: scene.clone(),
                },
            )
            .map_err(|error| ScriptError {
                output: String::new(),
                message: error.to_string(),
            })?;
            module.add("data", data).map_err(|error| ScriptError {
                output: String::new(),
                message: error.to_string(),
            })?;
            module
                .add("context", context)
                .map_err(|error| ScriptError {
                    output: String::new(),
                    message: error.to_string(),
                })?;
            module.add("ops", ops).map_err(|error| ScriptError {
                output: String::new(),
                message: error.to_string(),
            })?;

            let sys = PyModule::import(py, "sys").map_err(|error| ScriptError {
                output: String::new(),
                message: error.to_string(),
            })?;
            sys.getattr("modules")
                .and_then(|modules| modules.set_item("bpy", &module))
                .map_err(|error| ScriptError {
                    output: String::new(),
                    message: error.to_string(),
                })?;
            let mathutils = PyModule::new(py, "mathutils").map_err(|error| ScriptError {
                output: String::new(),
                message: error.to_string(),
            })?;
            mathutils
                .add("Vector", py.get_type::<MathVector>())
                .and_then(|_| mathutils.add("Euler", py.get_type::<MathEuler>()))
                .and_then(|_| mathutils.add("Quaternion", py.get_type::<MathQuaternion>()))
                .map_err(|error| ScriptError {
                    output: String::new(),
                    message: error.to_string(),
                })?;
            sys.getattr("modules")
                .and_then(|modules| modules.set_item("mathutils", mathutils))
                .map_err(|error| ScriptError {
                    output: String::new(),
                    message: error.to_string(),
                })?;

            let io = PyModule::import(py, "io").map_err(|error| ScriptError {
                output: String::new(),
                message: error.to_string(),
            })?;
            let stdout = io
                .getattr("StringIO")
                .and_then(|class| class.call0())
                .map_err(|error| ScriptError {
                    output: String::new(),
                    message: error.to_string(),
                })?;
            let stderr = io
                .getattr("StringIO")
                .and_then(|class| class.call0())
                .map_err(|error| ScriptError {
                    output: String::new(),
                    message: error.to_string(),
                })?;
            let old_stdout = sys.getattr("stdout").map_err(|error| ScriptError {
                output: String::new(),
                message: error.to_string(),
            })?;
            let old_stderr = sys.getattr("stderr").map_err(|error| ScriptError {
                output: String::new(),
                message: error.to_string(),
            })?;
            sys.setattr("stdout", &stdout)
                .and_then(|_| sys.setattr("stderr", &stderr))
                .map_err(|error| ScriptError {
                    output: String::new(),
                    message: error.to_string(),
                })?;

            let globals = PyDict::new(py);
            let code = CString::new(source).map_err(|error| ScriptError {
                output: String::new(),
                message: format!("Python source contains a null byte: {error}"),
            })?;
            let run_result = globals
                .set_item("__name__", "__blendcraft_script__")
                .and_then(|_| py.run(&code, Some(&globals), None));
            let output = stdout
                .call_method0("getvalue")
                .and_then(|value| value.extract::<String>())
                .unwrap_or_default();
            let error_output = stderr
                .call_method0("getvalue")
                .and_then(|value| value.extract::<String>())
                .unwrap_or_default();
            let _ = sys.setattr("stdout", old_stdout);
            let _ = sys.setattr("stderr", old_stderr);

            if let Err(error) = run_result {
                let traceback = PyModule::import(py, "traceback")
                    .and_then(|traceback| {
                        traceback.call_method1(
                            "format_exception",
                            (error.get_type(py), error.value(py), error.traceback(py)),
                        )
                    })
                    .and_then(|lines| lines.extract::<Vec<String>>())
                    .map(|lines| lines.concat())
                    .unwrap_or_else(|_| error.to_string());
                return Err(ScriptError {
                    output: format!("{output}{error_output}"),
                    message: traceback,
                });
            }

            let state = match Arc::try_unwrap(scene) {
                Ok(scene) => scene.into_inner().map_err(|_| ScriptError {
                    output: format!("{output}{error_output}"),
                    message: "Script scene state is unavailable".to_owned(),
                })?,
                Err(scene) => scene
                    .lock()
                    .map_err(|_| ScriptError {
                        output: format!("{output}{error_output}"),
                        message: "Script scene state is unavailable".to_owned(),
                    })?
                    .clone(),
            };
            Ok(state.result(format!("{output}{error_output}")))
        })
    }

    fn configure_bundled_python() -> Result<(), ScriptError> {
        static CONFIGURED: OnceLock<Result<(), String>> = OnceLock::new();
        CONFIGURED
            .get_or_init(|| {
                let executable = std::env::current_exe()
                    .map_err(|error| format!("Could not locate BlendCraft executable: {error}"))?;
                let executable_dir = executable
                    .parent()
                    .ok_or_else(|| "BlendCraft executable has no parent directory".to_owned())?;
                let home = executable_dir.join("python");
                if !home.is_dir() {
                    return Ok(());
                }
                if !has_standard_library(&home) {
                    return Err(format!(
                        "Bundled Python runtime is incomplete at {}",
                        home.display()
                    ));
                }
                std::env::set_var("PYTHONHOME", &home);
                #[cfg(target_os = "windows")]
                {
                    let mut paths = vec![
                        executable_dir.to_path_buf(),
                        home.clone(),
                        home.join("DLLs"),
                    ];
                    if let Some(existing) = std::env::var_os("PATH") {
                        paths.extend(std::env::split_paths(&existing));
                    }
                    let path = std::env::join_paths(paths)
                        .map_err(|error| format!("Could not configure Python DLL paths: {error}"))?;
                    std::env::set_var("PATH", path);
                }
                Ok(())
            })
            .clone()
            .map_err(|message| ScriptError {
                output: String::new(),
                message,
            })
    }

    fn has_standard_library(home: &Path) -> bool {
        if home.join("Lib/os.py").is_file() {
            return true;
        }
        home.join("lib")
            .read_dir()
            .ok()
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .any(|entry| entry.path().join("os.py").is_file())
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "python-scripting"))]
pub use embedded::execute;

#[cfg(all(not(target_arch = "wasm32"), not(feature = "python-scripting")))]
pub fn execute(_source: String, _snapshot: ScriptSnapshot) -> Result<ScriptResult, ScriptError> {
    Err(ScriptError {
        output: String::new(),
        message:
            "Python scripting is unavailable in this build; enable the `python-scripting` feature."
                .to_owned(),
    })
}

#[cfg(all(test, not(target_arch = "wasm32"), feature = "python-scripting"))]
mod tests {
    use super::{execute, ScriptObject, ScriptSnapshot};

    fn empty_snapshot() -> ScriptSnapshot {
        ScriptSnapshot {
            objects: Vec::new(),
            materials: Vec::new(),
            active: None,
        }
    }

    #[test]
    fn shared_example_updates_the_scene_and_prints_results() {
        let snapshot = ScriptSnapshot {
            objects: vec![ScriptObject {
                id: Some(12),
                name: "Cube".to_owned(),
                object_type: "MESH".to_owned(),
                primitive: "Cube".to_owned(),
                primitive_params: vec![2.0],
                material: None,
                location: [0.0; 3],
                rotation_euler: [0.0; 3],
                scale: [1.0; 3],
                selected: true,
            }],
            materials: Vec::new(),
            active: Some(0),
        };
        let result = execute(
            include_str!("../../../examples/blendcraft_blender_compat_test.py").to_owned(),
            snapshot,
        )
        .expect("shared Python example should run");

        assert_eq!(result.objects[0].name, "BC Compatibility Cube");
        assert_eq!(result.objects[0].location, [1.0, 2.0, 3.0]);
        assert_eq!(result.objects[0].material, Some(0));
        assert_eq!(result.materials[0].name, "BC Compatibility Material");
        assert_eq!(result.materials[0].diffuse_color, [0.2, 0.55, 0.9, 1.0]);
        assert!((result.objects[0].rotation_euler[2] - 0.25).abs() < 0.00001);
        assert!(result
            .output
            .contains("BlendCraft scripting compatibility test passed."));
        assert!(result.output.contains("BC Compatibility Material"));
    }

    #[test]
    fn cube_operator_adds_a_scene_object() {
        let result = execute(
            "import bpy\nbpy.ops.mesh.primitive_cube_add(size=2.0, location=(3.0, 0.0, 0.0))\nprint(bpy.context.object.name)\n".to_owned(),
            empty_snapshot(),
        )
        .expect("cube operator should run");

        assert_eq!(result.objects.len(), 1);
        assert_eq!(result.objects[0].name, "Cube");
        assert_eq!(result.objects[0].location, [3.0, 0.0, 0.0]);
        assert_eq!(result.active, Some(0));
        assert!(result.objects[0].selected);
        assert_eq!(result.output.trim(), "Cube");
    }

    #[test]
    fn mathutils_vector_and_context_scene_collection_work() {
        let snapshot = ScriptSnapshot {
            objects: vec![ScriptObject {
                id: Some(5),
                name: "Cube".to_owned(),
                object_type: "MESH".to_owned(),
                primitive: "Cube".to_owned(),
                primitive_params: vec![2.0],
                material: None,
                location: [0.0; 3],
                rotation_euler: [0.0; 3],
                scale: [1.0; 3],
                selected: true,
            }],
            materials: Vec::new(),
            active: Some(0),
        };
        let result = execute(
            "import bpy\nfrom mathutils import Euler, Quaternion, Vector\nobj = bpy.data.objects['Cube']\nassert obj is not None\nassert bpy.context.scene.objects[0].name == obj.name\nobj.location = Vector((1.0, 2.0, 3.0))\nobj.location.x += 0.5\nrotation = Euler((0.0, 0.0, 0.25)).to_quaternion()\nobj.rotation_quaternion = Quaternion(tuple(rotation))\nassert abs(obj.rotation_euler.z - 0.25) < 0.0001\nassert Vector((1.0, 0.0, 0.0)).cross(Vector((0.0, 1.0, 0.0)))[2] == 1.0\nprint(Vector((3.0, 4.0, 0.0)).length)\n".to_owned(),
            snapshot,
        )
        .expect("common mathutils and scene access should work");

        assert_eq!(result.objects[0].location, [1.5, 2.0, 3.0]);
        assert_eq!(result.output.trim(), "5.0");
    }

    #[test]
    fn common_mesh_operators_preserve_primitive_parameters() {
        let result = execute(
            "import bpy\nbpy.ops.mesh.primitive_uv_sphere_add(radius=1.5, segments=12, ring_count=8)\nbpy.ops.mesh.primitive_cylinder_add(radius=0.75, depth=3.0, vertices=16)\n".to_owned(),
            empty_snapshot(),
        )
        .expect("common mesh operators should run");

        assert_eq!(result.objects.len(), 2);
        assert_eq!(result.objects[0].primitive, "Sphere");
        assert_eq!(result.objects[0].primitive_params, [1.5, 12.0, 8.0]);
        assert_eq!(result.objects[1].primitive, "Cylinder");
        assert_eq!(result.objects[1].primitive_params, [0.75, 3.0, 16.0]);
        assert_eq!(result.active, Some(1));
    }

    #[test]
    fn materials_can_be_created_edited_and_assigned_to_objects() {
        let result = execute(
            "import bpy\nbpy.ops.mesh.primitive_cube_add()\nmaterial = bpy.data.materials.new(name='Red')\nmaterial.diffuse_color = (1.0, 0.0, 0.0, 1.0)\nmaterial.metallic = 0.7\nmaterial.roughness = 0.25\nbpy.context.object.data.materials.append(material)\nassert bpy.data.materials['Red'].name == 'Red'\nassert bpy.context.object.active_material.name == 'Red'\nprint(len(bpy.data.materials))\n".to_owned(),
            empty_snapshot(),
        )
        .expect("basic material creation and assignment should run");

        assert_eq!(result.materials.len(), 1);
        assert_eq!(result.materials[0].name, "Red");
        assert_eq!(result.materials[0].diffuse_color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(result.materials[0].metallic, 0.7);
        assert_eq!(result.materials[0].roughness, 0.25);
        assert_eq!(result.objects[0].material, Some(0));
        assert_eq!(result.output.trim(), "1");
    }

    #[test]
    fn script_errors_preserve_printed_output_and_traceback() {
        let error = execute(
            "print('before failure')\nraise ValueError('bad input')\n".to_owned(),
            empty_snapshot(),
        )
        .expect_err("invalid script should fail");

        assert!(error.output.contains("before failure"));
        assert!(error.message.contains("ValueError: bad input"));
    }
}
