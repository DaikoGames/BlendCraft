//! Platform-independent core: no windowing, no GPU. Compiles to WASM unchanged.
pub mod camera;
pub mod command;
pub mod constraints;
pub mod document;
pub mod environment;
pub mod kinds;
pub mod material;
pub mod mesh;
pub mod primitives;
pub mod project;
pub mod scene;

pub use camera::{FixedView, OrbitCamera};
pub use kinds::{helper_lines, helper_radius, CameraData, EmptyShape, LightData, LightType, ObjectKind};
pub use command::{Command, History};
pub use constraints::{Constraint, ConstraintKind};
pub use document::{AlignMode, AlignTo, Document, OriginMode, SameKind};
pub use environment::{sh_basis, EnvironmentMap};
pub use material::{AlphaMode, Material, TextureData};
pub use mesh::{Aabb, MeshData, MeshGroup, Vertex};
pub use primitives::Primitive;
pub use scene::{Collection, Object, Scene};
