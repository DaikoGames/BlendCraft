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
        P::IcoSphere { radius, subdivisions } => vec![*radius, *subdivisions as f32],
        P::Sphere { radius, segments, rings } => vec![*radius, *segments as f32, *rings as f32],
        P::Capsule { radius, height, segments, rings } => vec![*radius, *height, *segments as f32, *rings as f32],
        P::Cylinder { radius, height, segments }
        | P::Cone { radius, height, segments }
        | P::Prism { radius, height, segments }
        | P::Tube { radius, height, segments } => vec![*radius, *height, *segments as f32],
        P::Torus { major_radius, minor_radius, major_segments, minor_segments } => {
            vec![*major_radius, *minor_radius, *major_segments as f32, *minor_segments as f32]
        }
        P::Pyramid { size, height } => vec![*size, *height],
        P::CurveBezier { size, segments }
        | P::CurvePath { length: size, segments } => vec![*size, *segments as f32],
        P::CurveCircle { radius, segments } => vec![*radius, *segments as f32],
        P::Metaball { radius, subdivisions } => vec![*radius, *subdivisions as f32],
        P::Text { content, size, depth, .. } => vec![*size, *depth, content.len() as f32],
    };
    (primitive.name().to_owned(), params)
}


#[cfg(all(not(target_arch = "wasm32"), feature = "python-scripting"))]
mod embedded {
    use super::{ScriptError, ScriptMaterial, ScriptObject, ScriptResult, ScriptSnapshot};

    /// Escape a Rust string as a Python string literal (single-quoted).
    fn py_str(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('\'');
        for ch in s.chars() {
            match ch {
                '\\' => out.push_str("\\\\"),
                '\'' => out.push_str("\\'"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\x{:02x}", c as u32)),
                c => out.push(c),
            }
        }
        out.push('\'');
        out
    }

    fn py_f32(v: f32) -> String {
        if v.is_finite() {
            // Ensure a decimal so Python sees a float, not an int.
            let s = format!("{}", v);
            if s.contains('.') || s.contains('e') || s.contains('E') {
                s
            } else {
                format!("{s}.0")
            }
        } else if v.is_nan() {
            "float('nan')".to_owned()
        } else if v.is_sign_negative() {
            "float('-inf')".to_owned()
        } else {
            "float('inf')".to_owned()
        }
    }

    /// Emit the snapshot as a Python dict literal — no json module, no codecs.
    fn snapshot_to_python(snapshot: &ScriptSnapshot) -> String {
        let mut objects = String::from("[");
        for (i, o) in snapshot.objects.iter().enumerate() {
            if i > 0 {
                objects.push(',');
            }
            let id = match o.id {
                Some(id) => id.to_string(),
                None => "None".to_owned(),
            };
            let mat = match o.material {
                Some(m) => m.to_string(),
                None => "None".to_owned(),
            };
            objects.push_str(&format!(
                "{{'id':{id},'name':{name},'type':{ty},'primitive':{prim},'primitive_params':[{params}],'material':{mat},'location':[{lx},{ly},{lz}],'rotation_euler':[{rx},{ry},{rz}],'scale':[{sx},{sy},{sz}],'selected':{sel}}}",
                name = py_str(&o.name),
                ty = py_str(&o.object_type),
                prim = py_str(&o.primitive),
                params = o
                    .primitive_params
                    .iter()
                    .map(|p| py_f32(*p))
                    .collect::<Vec<_>>()
                    .join(","),
                lx = py_f32(o.location[0]),
                ly = py_f32(o.location[1]),
                lz = py_f32(o.location[2]),
                rx = py_f32(o.rotation_euler[0]),
                ry = py_f32(o.rotation_euler[1]),
                rz = py_f32(o.rotation_euler[2]),
                sx = py_f32(o.scale[0]),
                sy = py_f32(o.scale[1]),
                sz = py_f32(o.scale[2]),
                sel = if o.selected { "True" } else { "False" },
            ));
        }
        objects.push(']');

        let mut materials = String::from("[");
        for (i, m) in snapshot.materials.iter().enumerate() {
            if i > 0 {
                materials.push(',');
            }
            materials.push_str(&format!(
                "{{'name':{name},'diffuse_color':[{r},{g},{b},{a}],'metallic':{met},'roughness':{rough},'emissive':[{e0},{e1},{e2}]}}",
                name = py_str(&m.name),
                r = py_f32(m.diffuse_color[0]),
                g = py_f32(m.diffuse_color[1]),
                b = py_f32(m.diffuse_color[2]),
                a = py_f32(m.diffuse_color[3]),
                met = py_f32(m.metallic),
                rough = py_f32(m.roughness),
                e0 = py_f32(m.emissive[0]),
                e1 = py_f32(m.emissive[1]),
                e2 = py_f32(m.emissive[2]),
            ));
        }
        materials.push(']');

        let active = match snapshot.active {
            Some(a) => a.to_string(),
            None => "None".to_owned(),
        };
        format!("{{'objects':{objects},'materials':{materials},'active':{active}}}")
    }

    pub fn execute(source: String, snapshot: ScriptSnapshot) -> Result<ScriptResult, ScriptError> {
        use rustpython_vm as vm;
        use rustpython_vm::Interpreter;

        // Python dict literal — avoids json + codec stack (latin1 LookupError).
        let scene_lit = snapshot_to_python(&snapshot);

        // Build a self-contained Python program: shim + user code + result export.
        // User code is wrapped in try/except so we always get a full traceback.
        let shim = include_str!("bpy_shim.py");
        let indented_user: String = source
            .lines()
            .map(|line| {
                if line.is_empty() {
                    String::new()
                } else {
                    format!("    {line}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        // Minimal JSON serializer in pure Python (no import json / codecs).
        let program = format!(
            r#"{shim}
_BC = {scene_lit}
_EXC = None
_TB = None
try:
{indented_user}
except Exception as _e:
    import sys as _sys
    _EXC = "%s: %s" % (type(_e).__name__, _e)
    # Prefer traceback if available; otherwise synthesize a short message.
    try:
        import traceback as _tb
        _TB = _tb.format_exc()
    except Exception:
        _TB = _EXC

def _dumps(o):
    if o is None:
        return "null"
    t = type(o)
    if t is bool:
        return "true" if o else "false"
    if t is int:
        return str(o)
    if t is float:
        if o != o:
            return "null"
        return repr(o)
    if t is str:
        out = ['"']
        for ch in o:
            o_ = ord(ch)
            if ch == '"':
                out.append('\\"')
            elif ch == '\\':
                out.append('\\\\')
            elif ch == '\n':
                out.append('\\n')
            elif ch == '\r':
                out.append('\\r')
            elif ch == '\t':
                out.append('\\t')
            elif o_ < 0x20:
                out.append('\\u%04x' % o_)
            else:
                out.append(ch)
        out.append('"')
        return "".join(out)
    if t is list or t is tuple:
        return "[" + ",".join(_dumps(x) for x in o) + "]"
    if t is dict:
        parts = []
        for k, v in o.items():
            parts.append(_dumps(str(k)) + ":" + _dumps(v))
        return "{{" + ",".join(parts) + "}}"
    return _dumps(str(o))

_RESULT = _dumps({{
    "objects": _BC["objects"],
    "materials": _BC["materials"],
    "active": _BC.get("active"),
    "stdout": _stdout.getvalue(),
    "stderr": _stderr.getvalue(),
    "exc": _EXC,
    "traceback": _TB,
}})
"#,
            shim = shim,
            scene_lit = scene_lit,
            indented_user = indented_user,
        );

        let interp = Interpreter::with_init(Default::default(), |vm| {
            // Native modules written in Rust
            vm.add_native_modules(rustpython_stdlib::get_module_inits());
            // Pure-Python stdlib (json, traceback, …) via freeze-stdlib feature
            vm.add_frozen(rustpython_pylib::FROZEN_STDLIB);
        });

        interp.enter(|vm| {
            let scope = vm.new_scope_with_builtins();
            let code = match vm.compile(&program, vm::compiler::Mode::Exec, "<blendcraft>".to_owned()) {
                Ok(c) => c,
                Err(e) => {
                    return Err(ScriptError {
                        output: String::new(),
                        message: format!("SyntaxError: {e}"),
                    });
                }
            };
            if let Err(exc) = vm.run_code_obj(code, scope.clone()) {
                // Failure in the shim itself (before our try/except).
                use rustpython_vm::AsObject;
                let mut msg = format!("{exc:?}");
                // RustPython 0.4: get_arg takes only idx; class() needs AsObject in scope.
                if let Some(arg) = exc.get_arg(0) {
                    if let Some(s) = arg.downcast_ref::<vm::builtins::PyStr>() {
                        let type_name = exc.class().name().to_owned();
                        msg = format!("{type_name}: {}", s.as_str());
                    }
                }
                // Salvage any printed output
                let mut output = String::new();
                for name in ["_stdout", "_stderr"] {
                    if let Ok(obj) = scope.globals.get_item(name, vm) {
                        if let Ok(gv) = obj.get_attr("getvalue", vm) {
                            if let Ok(s) = gv.call((), vm) {
                                if let Some(ps) = s.downcast_ref::<vm::builtins::PyStr>() {
                                    output.push_str(ps.as_str());
                                }
                            }
                        }
                    }
                }
                return Err(ScriptError { output, message: msg });
            }

            let result_obj = scope.globals.get_item("_RESULT", vm).map_err(|_| ScriptError {
                output: String::new(),
                message: "script produced no _RESULT".to_owned(),
            })?;
            let result_str = result_obj
                .downcast_ref::<vm::builtins::PyStr>()
                .map(|s| s.as_str().to_owned())
                .ok_or_else(|| ScriptError {
                    output: String::new(),
                    message: "_RESULT is not a string".to_owned(),
                })?;

            #[derive(serde::Deserialize)]
            struct Wire {
                objects: Vec<ScriptObject>,
                #[serde(default)]
                materials: Vec<ScriptMaterial>,
                active: Option<usize>,
                #[serde(default)]
                stdout: String,
                #[serde(default)]
                stderr: String,
                #[serde(default)]
                exc: Option<String>,
                #[serde(default)]
                traceback: Option<String>,
            }
            let wire: Wire = serde_json::from_str(&result_str).map_err(|e| ScriptError {
                output: String::new(),
                message: format!("result parse: {e}"),
            })?;

            let output = wire.stdout + &wire.stderr;
            if let Some(exc) = wire.exc {
                let tb = wire.traceback.unwrap_or_default();
                // Prefer full traceback; fall back to the short "Type: message".
                let message = if tb.trim().is_empty() {
                    exc
                } else {
                    // traceback already ends with the exception line
                    tb
                };
                return Err(ScriptError { output, message });
            }

            Ok(ScriptResult {
                objects: wire.objects,
                materials: wire.materials,
                active: wire.active,
                output,
            })
        })
    }
}


#[cfg(all(not(target_arch = "wasm32"), feature = "python-scripting"))]
pub use embedded::execute;

#[cfg(all(not(target_arch = "wasm32"), not(feature = "python-scripting")))]
pub fn execute(_source: String, _snapshot: ScriptSnapshot) -> Result<ScriptResult, ScriptError> {
    Err(ScriptError {
        output: String::new(),
        message: "Python scripting is unavailable in this build; enable the `python-scripting` feature.".to_owned(),
    })
}

#[cfg(target_arch = "wasm32")]
pub fn execute(_source: String, _snapshot: ScriptSnapshot) -> Result<ScriptResult, ScriptError> {
    Err(ScriptError {
        output: String::new(),
        message: "Python scripting is not available on WebAssembly".to_owned(),
    })
}
