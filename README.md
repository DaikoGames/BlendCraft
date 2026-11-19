# BlendCraft

BlendCraft is an independent, local-first 3D creation application written in Rust. It is inspired by workflows found in Blender, but it is **not Blender**, does not contain Blender source code or artwork, and is not affiliated with or endorsed by the Blender Foundation. “Blender” is a trademark of the Blender Foundation.

BlendCraft is an early-stage project. Features described below are BlendCraft implementations and should not be assumed to have full Blender compatibility or parity. The project is designed to perform scene editing, rendering, painting, and simulation on the user's computer; its core workflows do not require sending scene data to a remote service.

## Project status

This README is the project progress record. A feature is marked **Finished** only when the implementation exists in the project. **Partially finished** means that a usable foundation exists but important behavior or edge cases are still missing. **Not finished** means it is not implemented or is only a placeholder.

### Stage 1 — quick improvements

| Feature | Status | Current implementation / remaining work |
|---|---|---|
| Basic Edit Mode selection | **Finished** | Vertex, edge, and face selection; click selection; Shift-click toggling; deselection; selection-domain switching; selected components can be transformed. |
| Flip Normals | **Finished** | Flips all faces when nothing is selected, or faces touched by the current vertex, edge, or face selection. |
| Merge at Center | **Finished** | Merges selected vertices to their median/center point and removes degenerate triangles. |
| Merge by Distance / Weld | **Finished** | `Shift+M` welds coincident selected vertices only; unselected vertices are left alone. |
| Smooth Vertices | **Finished** | Selected vertices use a one-pass Laplacian smooth with unique adjacent neighbors. Advanced iterative and volume-preserving smoothing are future enhancements. |
| Shrink/Fatten | **Finished** | Moves selected vertices along their averaged mesh normals; available through the toolbar/tool drag and `Alt+S`. |
| Texture Paint Erase | **Finished** | Eraser strokes restore the original texture pixels captured on entering Paint Mode, including user-loaded images. |
| Render output format | **Finished** | Render Image supports PNG, JPEG, and OpenEXR output; the selected format controls the save dialog filter and file extension. OpenEXR is written from the available 8-bit render buffer, so it does not add high-dynamic-range render data. |
| Fix Stage 1 Edit Mode operator issues | **Finished** | Selection-sensitive normal flipping, selected-only welding, duplicate-neighbor handling in smoothing, and coincident-vertex selection toggling have been corrected. Other topology operators remain simplified and are tracked in later stages. |

**Stage 1 implementation is complete.** The source-level changes are in place. Build and runtime verification is still pending in this environment because the Rust toolchain is not installed here. More advanced smoothing controls and topology-safe modeling remain later-stage work, not unfinished Stage 1 requirements.

### Stage 2 — painting and attribute foundations

| Feature | Status |
|---|---|
| Vertex Paint mode and vertex-color attributes | **Finished** | Per-vertex RGBA colors, Vertex Paint mode, Draw/Soften/Blur/Smear brushes. |
| Vertex Paint brush tools | **Finished** | Draw, Soften, Blur, Smear. |
| Editable vertex groups | **Finished** | Named groups with per-vertex weights; add/remove/active selection. |
| Weight Paint mode and weight visualization | **Finished** | Weight Paint mode, heat-map visualization, Draw/Blur/Average/Smear, optional normalize. |
| Weight normalization and practical weight editing | **Not finished** |
| Expanded Texture Paint brushes and image workflow | **Partially finished** — Draw, Erase, Soften, Smear, and Fill; eraser restores entry snapshot. Layers/masks still incomplete. |

### Stage 3 — intermediate modeling and animation tools

| Feature | Status |
|---|---|
| UV editing, projections, and basic island packing | **Partially finished** |
| UV seams, pinning, and advanced unwrap tools | **Not finished** |
| Sculpt brushes | **Partially finished** — a collection of basic brushes exists; masks, face sets, remeshing, and advanced brush behavior are incomplete. |
| Shader node editor | **Partially finished** — node editing and a supported real-time shader subset exist; unsupported nodes use limited fallback behavior. |
| Geometry Nodes | **Partially finished** — a small procedural graph evaluator exists, not a complete Blender-compatible system. |
| Armature Pose Mode and toolbar | **Partially finished** — bones and basic pose transforms exist; advanced rigging and constraints are incomplete. |
| Animation workspace / Dope Sheet | **Partially finished** — basic transform keyframes and timeline editing exist; Graph Editor, NLA, and drivers are not implemented. |

### Stage 4 — advanced modeling and rigging

| Feature | Status |
|---|---|
| Proper vertex, edge, and face extrusion | **Partially finished** — current operations are simplified and do not provide full polygon-mesh behavior. |
| Inset, Bevel, Loop Cut, Edge Slide, Spin | **Partially finished** — basic approximations exist; robust topology-aware behavior remains. |
| Subdivision surfaces | **Partially finished** — basic midpoint subdivision exists; production subdivision surfaces are not implemented. |
| Boolean operations | **Partially finished** — current implementation is simplified and is not robust general-purpose CSG. |
| Advanced UV unwrapping | **Not finished** |
| Armature constraints and advanced rigging | **Not finished** |
| Graph Editor and advanced animation tools | **Not finished** |
| Full compositor workflow | **Partially finished** |

### Stage 5 — core architecture and parity work

| Feature | Status |
|---|---|
| Robust polygon topology with shared vertices, edges, faces, and n-gons | **Not finished** — the current mesh representation is triangle-based. |
| Topology-safe editing and attribute preservation | **Not finished** |
| Advanced modifiers and geometry evaluation | **Partially finished** |
| Expanded Geometry Nodes | **Partially finished** |
| Advanced sculpting and remeshing | **Not finished** |
| Broader rendering, material, and animation compatibility | **Partially finished** |
| Simulation and workflow parity with Blender | **Partially finished** |

## Build and run

A Rust toolchain compatible with the workspace's declared Rust version is required.

```sh
cargo run --release -p forge-app
cargo run --release -p forge-app -- model.glb
cargo test -p forge-core
```

The desktop application binary is named `forge3d`. The workspace includes separate crates for the application, core scene/editing logic, rendering, file I/O, and video output.

## Main workflows currently implemented

### Scene and object editing

- Blender-style startup scene with a Cube, Sun Light, and Camera.
- Add mesh primitives, cameras, lights, empties, and image planes.
- Object selection, transforms, parenting, undo/redo, and `.forge` project save/load.
- Edit Mode with vertex, edge, and face selection; component transforms; delete; and a growing set of mesh operators.
- Non-destructive modifier stack with basic Subdivision, Array, Mirror, Solidify, Bevel, and Boolean implementations.

The mesh data and most Edit Mode operators are triangle-based. Some operations are approximations and may not preserve complex topology. The project still needs a robust shared vertex/edge/face topology model before it can provide reliable n-gon modeling and full BMesh-style behavior.

### UV Editing and Texture Paint

UV Editing provides a 2D UV editor alongside a 3D view, with basic selection and transforms, projections, packing, and save/load/export workflows. Texture Paint provides a 2D painting canvas and a 3D model view, basic brush settings, draw and erase tools, and local image saving.

Texture Paint requires usable UV coordinates. The eraser restores the texture snapshot captured when Paint Mode is entered, so it works with an existing loaded image as well as a generated paint atlas. Advanced brushes, masks, layers, and a complete image-management workflow are not implemented.

### Shading and materials

The Shading workspace includes an editable node graph, node links, movement, resizing, and graph persistence. A supported subset of nodes is compiled to a real-time WGSL shader; other graphs may use a CPU UV-bake fallback or remain unsupported. This is a BlendCraft renderer, not Blender EEVEE/Cycles parity.

### Rendering and output

The application includes a Workbench-style viewport, a real-time PBR renderer, and a progressive CPU path tracer. Render Image uses the Output settings for dimensions and supports PNG, JPEG, and OpenEXR file output. JPEG drops alpha. OpenEXR currently receives values converted from the 8-bit render buffer; it is an EXR container, not a true high-dynamic-range render pipeline.

Rendering features, image formats, and GPU behavior still require testing across supported operating systems and graphics hardware.

### Compositing

The Compositing workspace supports a basic image-node graph with Render Layers, Composite, Viewer, Brightness/Contrast, Exposure, Blur, Glare, and Invert. Image operations currently use 8-bit RGBA CPU buffers. Blender's full node catalog, render passes, masks, multi-input compositing, and full color management are not implemented.

### Animation

The timeline supports frame range, frame rate, playback, and basic object transform keyframes. A simple Dope Sheet view is available. A full Graph Editor, NLA system, drivers, and complete animation data workflows are not implemented.

### Armatures

Basic armature creation/editing, bones, Pose Mode transforms, and automatic skin weights exist. Automatic weights use a simplified distance-based method rather than Blender's heat weighting. Advanced constraints, IK, editable vertex groups, and a complete Weight Paint workflow are not implemented.

### Physics and simulation

- Rigid-body simulation uses the embedded Rapier3D library.
- Cloth uses a local XPBD-based solver.
- Fluid uses a local CPU Position Based Fluids implementation.

These are independent implementations and are not expected to reproduce Blender's physics solvers exactly. Simulation and cross-platform behavior need broader validation.

### Scripting

The Scripting workspace embeds RustPython and provides a limited Blender-like Python bridge. The supported API subset includes basic object access, transforms, selection, primitive creation, and simple material operations. It is not full `bpy` compatibility; unsupported calls are not silently emulated.

### File formats

The project supports its native `.forge` format and has import/export paths for formats including glTF/GLB, OBJ, FBX, and Alembic. Format coverage varies: some attributes and object types are skipped or simplified. Native `.blend` import/export is not implemented.

## Navigation controls

- Middle mouse drag: orbit.
- Shift + middle mouse / right mouse drag: pan.
- Scroll: zoom.
- Left drag away from a gizmo: orbit; Shift + left drag pans.
- Ctrl + left drag: orbit (Unity-style).
- `0` or View → Camera View: look through the active camera.

## Edit Mode controls

- `Tab`: toggle Edit Mode.
- `1`, `2`, `3`: Vertex, Edge, and Face selection domains.
- Click: select a component.
- Shift-click: add or toggle a component in the selection.
- `A`: select all; `Alt+A`: deselect all.
- `G`, `R`, `S`: move, rotate, and scale selected components.
- `M`: merge selected vertices at center.
- `Shift+M`: merge selected coincident vertices by distance.
- `Alt+S`: Shrink/Fatten.
- `N`: Flip Normals.

## Known limitations

- No full Blender/BMesh-compatible mesh topology or complete Edit Mode parity.
- No native `.blend` import/export.
- Vertex Paint and Weight Paint are implemented with basic brushes; advanced masks/layers are not.
- Advanced UV seams, pinning, and unwrap tools are not implemented.
- Sculpt masks, face sets, and advanced remeshing are not implemented.
- Full animation editors, NLA, and drivers are not implemented.
- The shader, compositor, modifier, physics, and file-format implementations are subsets.
- Cross-platform builds and hardware-specific rendering still need validation on each target.

## Development notes

When a feature is completed, update the relevant row in **Project status** to **Finished** and remove its remaining-work description. Keep this README synchronized with the actual source code; do not mark a feature finished merely because a UI control or placeholder exists.
