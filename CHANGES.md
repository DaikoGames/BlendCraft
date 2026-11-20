## Step 2: glTF/GLB skeletal animation import

- Imports glTF animation channels targeting skin joints: translation, rotation and scale.
- Converts sampled node transforms into pose deltas relative to each joint's imported rest transform.
- Stores tracks separately per named clip, so choosing another action does not merge or overwrite its keys.
- Evaluates the active clip during timeline scrubbing and playback, updates armature pose channels, and re-applies skin deformation.
- Handles the central value samples of glTF CUBICSPLINE output accessors. Tangent-based cubic interpolation and exact STEP interpolation are not yet reproduced; tracks use BlendCraft's existing key interpolation.
- Fixes scene merging so imported object parents, mesh-to-armature skin links, object animation tracks, and bone tracks are remapped to the new document object IDs instead of being discarded.
- Scope: glTF/GLB joint animation. FBX animation import/export and skeletal animation export are not part of this step.


## Import fixes: textures, hierarchies, bones

- **FBX**: textures are now imported. Embedded images (binary `Video/Content`, ASCII base64) and external files (searched next to the model, in `textures/`, `<name>.fbm/`, case-insensitively, by bare file name) are decoded and wired to base color, normal, emissive and occlusion slots through the `OP` connections. The grey diffuse default no longer darkens a textured material.
- **FBX**: normals and UVs honour `MappingInformationType` / `ReferenceInformationType` (`ByPolygonVertex`, `ByVertice`, `IndexToDirect`...). Previously UVIndex was read per vertex, which scrambled UVs on almost every file.
- **FBX**: Model-to-Model parent links are followed, so meshes parented to bones / nulls / armatures land in the right place (parent chain is baked into the object transform). `PreRotation`, `PostRotation` and `RotationOrder` are honoured, Euler order was wrong for multi-axis rotations, Z-up files are converted to Y-up, `UnitScaleFactor` no longer gets overridden by `OriginalUnitScaleFactor`.
- **FBX**: multi-material meshes get one group per material slot; blend-shape "Shape" geometries are no longer imported as junk meshes; one unreadable node no longer aborts the whole file.
- **OBJ**: `.mtl` diffuse / normal textures and dissolve are imported.
- **glTF/GLB**: skinned meshes are posed at bind pose from their joints (meshes under a skeleton), triangle strips/fans are converted, files without a `scene` still load.
- **.unitypackage**: textures and `.mtl` files stored as separate assets in the package are found by file name for FBX and OBJ models.
- Images: TGA and BMP are decoded (also for "Image as Plane").

## X Mirror (butterfly) + Pose bone connections

- **X Mirror**: toggle via header **🦋 X Mirror** or **M**. Mirrors Edit Mode transforms, Sculpt strokes, Armature Edit, and Pose bone transforms across the local X axis (L/R bone names `.L`/`.R` supported).
- **Pose Mode connections**: hierarchical pose evaluation snaps connected child heads to the parent tip so chains stay attached while posing.


## Stage 2 paint systems (local / offline)

- **Vertex colors** on `Vertex` + `MeshData` vertex groups (`VertexGroup`).
- **Vertex Paint mode** (`Ctrl+Shift+V`): Draw / Soften / Blur / Smear brushes.
- **Weight Paint mode** (`Ctrl+Shift+W`): Draw / Blur / Average / Smear, heat-map visualization, optional weight normalization.
- **Editable vertex groups**: add/remove/set active on the active mesh.
- **Texture Paint brushes**: Soften, Smear, Fill in addition to Draw/Erase.
- All painting runs fully offline (no server APIs).

# Changes: Cycles engine + cloth

Please run:

    cargo test -p forge-core -p forge-render
    cargo build --release

## Additional Edit Mode operators
- **Subdivide**: subdivides selected triangles into four, caches shared edge midpoints to keep adjacent selected faces connected, and interpolates UV coordinates. Shortcut: `Ctrl+Shift+D`.
- **Flip Normals**: reverses selected faces when Face Select has a selection, or the complete mesh otherwise. Shortcut: `N`.
- Added core unit tests for triangle subdivision and face winding reversal.

## Fixes (this pass)
- **Cloth bake freeze**: baking no longer runs the whole timeline on the UI thread.
  Object Physics → Cloth → **Bake** now uses a progressive `ClothBaker` (same pattern as
  fluid): ~30 ms of simulation per frame, progress bar, Cancel. Soft body / particles and
  “Bake All” remain synchronous for now.
- **Cycles shading textures**: the path tracer stores UVs, samples `base_color_tex`, and
  **auto-bakes** any material `shader_graph` (256², cached) when no base-color texture is
  set — so live EEVEE graphs appear in Cycles without a manual Bake click.
- **Shading / Workbench crash**: graph pipeline creation is wrapped so a bad generated WGSL
  falls back to the default shader instead of panicking the process.

## Files
- crates/render/src/cycles.rs   – path tracer + UV/texture sampling + graph auto-bake
- crates/render/src/lib.rs      – safer graph pipeline creation
- crates/core/src/cloth.rs      – fabric presets + XPBD solver + progressive `ClothBaker`
- crates/core/src/simulation.rs – cloth delegates to cloth.rs; Free Bake restores rest pose
- crates/core/src/project.rs    – format FRGA; FRG7–FRG9 cloth settings migrated
- crates/core/src/lib.rs        – exports ClothBaker, FabricType, ClothPin, FabricPreset
- crates/app/src/ui.rs          – progressive cloth bake UI + shared cloth panel
- README.md                     – documentation

## One resolution for the 3D view and renders
- **Resolution X / Y / %** (Output → Resolution) is now the single size setting.
  - **Render Image** and **Render Animation** always use it (they no longer force the panel size).
  - **Cycles 3D view** is path-traced at that size (exactly in camera view, same pixel density
    outside camera view). **Pixel Size** (1×/2×/4×/8×) only divides it for a faster view.
  - **Camera view** letterboxes to the same aspect, so what you see is what F12 renders.
- Removed **Max Size** and **Match viewport size** (they were a second, conflicting resolution).
- Fixed: viewport samples / pixel size defaults were only set once the Render panel had been
  drawn, so the Cycles view was noisy/low-res until you opened it.

## Sculpt mirror performance + mode UI (2026-11-19)

- **Sculpt Mirror XYZ lag fixed**: `apply_brush` no longer recursively re-applies the stroke for every mirror combination (which caused multi-second freezes on dense meshes). Centers are collected once and influences are applied in a single vertex pass; normals are recomputed only once at the end.
- **Mode selector**: Blender-style ComboBox at the start of the header. Primary modes (Object / Edit / Sculpt / Pose) stay at the top; Vertex Paint, Weight Paint and Texture Paint appear under a separator, matching the classic mode menu layout.
- Stage 2 weight normalization marked Finished in the README (normalize_vertex_groups + WeightPaintSettings.normalize already present and wired).
