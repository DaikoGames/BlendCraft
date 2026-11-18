# BlendCraft
This project was made with the Help of Grok and Claude
BlendCraft is an open-source clone of Blender, written from scratch in Rust.
It is an independent project: it contains no Blender source code or artwork, and it is
not affiliated with or endorsed by the Blender Foundation. "Blender" is a trademark of
the Blender Foundation.

BlendCraft is early and incomplete. If you want the real, mature 3D suite,
please visit and support the original at https://www.blender.org.


Rust 3D suite for game assets (formerly Forge3D).

- Step 1: workspace, viewport, glTF viewer
- Step 2: PBR materials/textures, HDRI lighting, view modes
- Step 3: egui UI (menu bar, outliner, properties), parametric primitives, selection,
  transforms, materials, undo/redo, project save/load (.forge), import glTF/GLB/OBJ, export GLB/OBJ
- Step 7+: Blender-like defaults (Cube + Sun + Camera), auto transform gizmo, improved outliner parenting,
  sky/IBL off by default, branding update

## Run
```
cargo run --release -p forge-app
cargo run --release -p forge-app -- model.glb
cargo test -p forge-core
```

## Workflow
1. On start you get a default Cube, Sun Light and Camera (Blender-style).
2. The Move gizmo (arrows) is active by default so selected objects show transform handles immediately.
3. Add more objects via Add > Mesh / Light / Camera / Empty. Tweak parameters, transform and material in the Properties panel.
4. Combine objects to build a model. Click objects in the viewport to select them. Right-click in the Outliner for Rename / Delete / Parenting / etc.
5. File > Save (.forge) to keep the project, File > Export > glTF Binary to use it in Unity/Godot.

## Shader graphs
The renderer compiles supported connected graphs into a WGSL fragment shader and evaluates them
per rendered surface fragment; compiled pipelines are cached, and parameter changes update a
uniform buffer instead of recompiling. The current GPU subset includes Value, RGB, texture
coordinates, Mapping, Noise and White Noise (1D-4D), Voronoi, Brick, Checker, Gradient, Magic,
Gabor, Wave, Math, Color Ramp, Mix, Combine/Separate XYZ and Color, Principled BSDF, Emission, and
Background. Graphs using other nodes continue through the asynchronous CPU UV-bake path, and the
renderer logs which node prevented real-time compilation. In the real-time path, unconnected
texture vector inputs use per-fragment object-space positions, so meshes without UV maps still show
procedural patterns; explicit coordinate links take precedence. The CPU bake fallback still relies
on UVs. Use **Bake** when an image texture is needed for export.

Color Ramp supports up to 32 stops, Blender's RGB/HSV/HSL color modes, hue direction, and
Linear/Ease/B-Spline/Cardinal/Constant interpolation. Stop handles can be selected and dragged;
position, alpha, color, and stop count are editable. Ramps using HSV/HSL or more than four stops use
the CPU bake path to stay within the real-time uniform-buffer limit. Shader nodes can be moved
by dragging their body or title and resized from their edges; their layout is saved with the material.
Drag empty canvas to pan, or use middle mouse anywhere in the canvas. Drag an output socket to an
input socket to connect them; right-click a visible wire to remove it. Socket clicks can also be
used to select the two ends of a connection.

Wave exposes Bands/Rings, direction, profile, and distortion detail controls; Brick exposes
offset/frequency and squash/frequency settings; Magic exposes its iteration depth. White Noise
matches Blender's 1D-4D dimension choices and has no Scale or Detail input: use a Mapping node to
change its feature size.

This is an EEVEE-style real-time implementation, not exact Blender EEVEE parity. Procedural
functions, texture mode details, and lighting are BlendCraft implementations. Image Texture, Bump,
Normal Map, other shader closures, Cycles-only behavior, and some coordinate semantics still use
the baked fallback or are unsupported. Blender's EEVEE and Cycles are different rendering engines,
so a single mode cannot match both. Run `cargo run --release -p forge-app`, open the Shading
workspace, and try the **Noise Demo** (Noise → Color Ramp → Principled) to see the live GPU path.
See the [Blender Shader Nodes manual](https://docs.blender.org/manual/en/latest/render/shader_nodes/index.html)
for Blender's reference behavior.

## Compositing
The **Compositing** workspace captures the current frame and evaluates a saved image node graph
off the UI thread. The initial graph passes Render Layers to Composite and Viewer. Supported nodes
are Brightness/Contrast, Exposure, Blur, Glare, and Invert; render results can be previewed and
saved as PNG. Nodes can be added, moved, connected by dragging their sockets, and deleted; pan the
canvas by dragging empty space or middle-drag, zoom with the wheel, and right-click a wire to remove
it. The graph is stored in `.forge` projects. Projects are stored as FRGA; older formats (FRG3-FRG9) remain readable, and cloth settings from
FRG7-FRG9 files are migrated.

This is a functional starter compositor, not full Blender compositor parity. It currently uses
8-bit RGBA CPU image operations and does not implement Blender's full node catalog, multi-input
compositing, masks, multilayer render passes, or color-management pipeline.

## Rigid-body physics

BlendCraft includes a local, Rust-native 3D rigid-body simulation powered by the embedded
Rapier3D library. It does not send scene data to a server and does not call an online physics
service; simulation, playback and baking are performed on the user's computer. Rapier is a
build-time Rust dependency, not a runtime network service.

In **Object Mode**, select a mesh and open the **Physics** tab in the Properties editor. Add an
**Active** or **Passive** rigid body. Object settings include dynamic/animated behavior, mass,
collision shape (Convex Hull, Box, Sphere, Capsule, Cylinder, Cone or Mesh), friction, bounciness,
collision margin, linear/angular damping, deactivation and continuous collision detection.
Mesh collision is intended for static/passive bodies; active mesh bodies use a convex hull.

The **Scene** properties tab exposes the Rigid Body World settings: gravity, simulation speed,
substeps, solver iterations and cache range. Its Cache controls bake transforms for the configured
frame range and delete a bake. The constraint editor supports Fixed, Point, Hinge and Slider joints between two objects
that both have rigid bodies, with collision disabling, anchors, axes and optional limits. Animated
rigid bodies are driven by object keyframes and can affect simulated bodies. Playback advances the
solver frame-by-frame, including catch-up frames, and baked frames are used when a cache exists.

This is a functional rigid-body foundation, not full Blender physics parity. It does not yet
implement Bullet-identical simulation, Blender's full rigid-body constraint catalog (including
piston, generic spring and motor variants), soft-body fidelity,
fracture, or a complete Blender-compatible physics API. Physics behavior and cross-platform
builds still need testing on each target OS and architecture.

## Cloth
Select a mesh, open **Physics**, add **＋ Cloth** and press **Bake**. Pick a **Fabric** to set every
physical value at once; any value can still be edited afterwards (the fabric then switches to
*Custom*).

| Fabric  | Mass kg/m² | Tension N/m | Bending mN·m | Notes |
|---------|-----------:|------------:|-------------:|-------|
| Silk    | 0.06 | 600  | 0.05 | floats, flutters, tiny wrinkles |
| Cotton  | 0.20 | 800  | 1    | everyday fabric |
| Linen   | 0.25 | 1500 | 5    | crisper folds |
| Wool    | 0.35 | 400  | 3    | heavy, stretchy, grippy |
| Denim   | 0.40 | 2000 | 15   | stiff, wide folds |
| Leather | 0.80 | 3000 | 200  | very stiff, barely stretches |
| Rubber  | 1.20 | 120  | 60   | elastic, high friction |

The solver is XPBD with sub-stepping: resolution-independent stretch/compression stiffness,
dihedral-angle bending driven by a real bending rigidity, aerodynamic drag (light fabrics float,
heavy ones fall fast) and a hard strain limit. Cloth collides with every other visible mesh
(floors, spheres, characters), supports optional self collision, and reacts to **Wind, Force,
Vortex and Turbulence** force-field objects (Add > Force Field). Pin the **Boundary**, **Top Edge**
or **Top Corners**; raise **Pin Depth** to clamp several rows so stiff fabrics stick out of their pins.

Low-poly meshes (like the default plane) are refined automatically while baking (**Sim Edge
Length**, default 4 cm; 0 disables it), so cloth drapes without manual subdividing. The baked cache
holds the refined mesh; **Free Bake** restores the rest pose. Colliders are taken at the current
frame (animated colliders are not followed yet) and pins do not follow animated transforms.

## Cycles viewport
The Cycles engine is a progressive CPU path tracer: a binned-SAH BVH, all CPU cores, up to 1920 px
on the long side, a quick low-resolution preview while the camera moves (the resolution adapts to
the machine speed), smooth shading normals, GGX specular with correct roughness/metallic, soft
shadows, spot cones, emissive surfaces and per-group materials. Image textures and
transparency/refraction are not traced yet.

## Water physics (Fluid)
Select a mesh, open **Physics > Fluid** and choose a role, as in Blender:

- **Domain** - a box that bounds the simulation. Settings: resolution divisions, particles per cell,
  viscosity, surface tension, gravity (scene or custom), time scale, substeps, solver iterations,
  particle budget, surface-mesh generation (scale, smoothing), open/closed walls and the cache range.
- **Flow** - emits water from a closed mesh: *Geometry* (initial volume), *Inflow* (continuous, with a
  frame range) or *Outflow* (sink). Optional initial velocity and plane initialization (fill below an open plane).
- **Effector** - a collision mesh. Keyframed, rigid-body-baked or mesh-cached effectors push the water.

**Bake Water** simulates the cache range (a little each UI frame, with a progress bar and Cancel) and stores one
surface mesh per frame on a "Fluid Mesh" object. It plays back on the timeline and exports with Alembic.
The solver is a local CPU Position Based Fluids implementation (parallelised with std threads), not Mantaflow/FLIP:
the workflow matches Blender, the exact motion does not. Spray/foam/bubble particles, viscosity presets, guides,
obstacles with surface friction and gas/smoke are not implemented. The old settings-only Fluid component is converted
to a Domain when an older project is opened.

## Alembic (.abc)
File > Import accepts `.abc` (also by drag and drop); File > Export > Alembic writes the scene or the selection.
Import reads PolyMesh/SubD objects and Xform hierarchies, including per-frame meshes whose topology changes
(water surfaces). Animated meshes become a **Mesh Sequence Cache** on the object (Physics tab: offset, speed, remove)
and the timeline range is set to the cache. Animated transforms become keyframes. Polygons are triangulated; normals
are recomputed; UVs, cameras, curves, points and custom attributes are skipped. Export samples animation, rigid-body
bakes and mesh sequences over the timeline range (static scenes are written as one sample). The Ogawa container and
Alembic layer are implemented in pure Rust (`crates/io/src/alembic.rs`). Property headers respect the official
variable-width integer encoding (size hint), so files written by Blender and other Alembic 1.7/1.8 tools import
correctly. The exporter follows the structure of a real Blender-written file (no hash after property headers, a hash
block after every object, empty dimension blocks for rank-1 arrays, uniform time sampling). HDF5-based archives from very old Alembic versions are not supported. Large water caches are held in
memory, one mesh per frame.

## Scripting
The **Scripting** workspace runs Python in an embedded **RustPython** interpreter from a
background worker, so scripts do not block the UI. BlendCraft itself and its scene bridge remain
Rust; the small `bpy`-named compatibility module is implemented natively in Rust (plus a thin
pure-Python helper layer) and is not copied from Blender.

RustPython is a pure-Rust Python 3 interpreter. Because it does not depend on a system CPython
or on platform-specific CPython development libraries, the same binary builds and runs on
Windows, Linux, macOS, FreeBSD and any other target that the Rust toolchain supports. No separate
Python installation and no bundled `python/` directory are required.

The `python-scripting` Cargo feature is enabled by default. Disable it only when you intentionally
want a build without scripting support.

The implemented subset includes object lookup (name/index lookup, iteration, membership),
context scene/active/selected objects, object names and transforms, selection methods, and
`mathutils.Vector`, `Euler`, and `Quaternion` basics. Supported operators are object select-all
(select/deselect/invert) and mesh add for cube, plane, UV sphere, ico sphere, cylinder,
cone, torus, and monkey. Basic materials support lookup/creation, name, diffuse color,
metallic/roughness, and one material slot per object.

This is not full `bpy` compatibility: mesh editing, multiple material slots, node trees, modifiers,
collections/linking, cameras/lights creation, animation, rendering, UI context, Blender-specific
modules, and many other Blender systems are not implemented. Unsupported API calls fail rather
than being silently emulated. Successful scene changes are applied together as one undoable
operation.

## Navigation
- Middle mouse drag: orbit
- Shift + Middle / Right drag: pan
- Scroll: zoom
- Left drag (when not on a gizmo handle): orbit (or pan with Shift)
- Ctrl + Left drag: orbit (Unity-style)
- View > Camera view (or 0) to look through a camera object

## Shadows & render
- **Real-time shadows**: on by default. 2048² shadow map from the primary Sun (preferred) or Spot light, with 3×3 PCF soft shadows. Toggle and adjust softness under **View**.
- **Render Image**: **View > Render Image…** captures the current 3D view (no UI) to a PNG you choose.

## Modifiers
Objects have a classic non-destructive **modifier stack** (Properties → Modifiers, or Tools → Modifiers):

- **Subdivision Surface** – simple midpoint levels (viewport + render levels)
- **Array** – count, relative/absolute offset, optional merge
- **Mirror** – X/Y/Z axes, optional merge at the plane
- **Solidify** – thickness + offset, side walls on boundary edges
- **Bevel** – amount, segments, angle limit (chamfer-style)
- **Boolean** – union / difference / intersect against another mesh object

The stack is evaluated before Geometry Nodes. Apply order can be changed with ▲/▼; each modifier can be toggled for viewport and render independently. Old projects load with an empty stack.

## Not yet there
Full Edit Mode parity (many BMesh operators), Curves/NURBS editing depth, native .blend import/export.
Point-light omnidirectional shadows, cascaded shadow maps for large scenes.
Vertex/Weight paint, full animation editors (NLA/drivers), Game-dev helpers (LODs, collider gen, lightmap UVs).

## Gizmo
Select an object (Move tool is default), then drag the colored handles. Press G / R / S for modal move/rotate/scale, Q returns to select mode. Every drag is one undo step.

## Object types (Add menu)
- Mesh: Plane, Cube, Circle, UV Sphere, Ico Sphere, Cylinder, Cone, Torus, Grid, Capsule (parameters stay editable)
- Empty: Plain Axes, Arrows, Single Arrow, Circle, Cube, Sphere, Cone
- Light: Point, Sun, Spot (real-time; up to 8 lights). Directional/Spot shadows supported.
- Camera: Perspective / Orthographic. Press 0 (or View > Camera view) to look through it.
- Image as Plane: textured upright quad from a PNG/JPG
- GLB export includes cameras, lights (KHR_lights_punctual) and empties.
- Greyed out in the menu (need bigger systems first): Curves, Surface, Metaball, Text, Volume, Grease Pencil,
  Armature, Lattice, Area light, Light Probe, Speaker, Force Field, Collection Instance.

## Object mode (complete set)
- Selection: click, Shift+click, Box / Lasso / Circle tools, Select All / None / Invert / Random,
  Select Children / Parent / Similar (type, collection, material, linked).
- Transform: gizmo (Move / Rotate / Scale) and modal G / R / S with axis constraints, plane constraints,
  numeric input, snapping (increment or vertex), pivot points, transform orientations (Global / Local / View),
  3D cursor (Shift+Right click).
- Duplicate (Shift+D), linked duplicate (Alt+D), single user, delete, hide / show.
- Parenting (Ctrl+P or Outliner context menu → Set Parent / Clear Parent), collections (flat), outliner tree with filter, rename (F2 / double click).
- Set origin, geometry to origin, apply / clear transforms, align objects, snap selection / cursor.
- Join (Ctrl+J, mixed materials become slots) and Separate (by loose parts, by material).
- Constraints: Copy Location / Rotation / Scale / Transforms, Track To, Limit Location / Rotation / Scale,
  Limit Distance, Floor.
- Help > Keyboard Shortcuts (or F1) lists every live shortcut and is generated from the key table.

## Export
File > Export has glTF Binary and Wavefront OBJ, each also as "selected only".
- OBJ does not support lights/cameras (format limitation); use GLB for those.
- GLB uses KHR_lights_punctual; intensity is stored in candela / lux units. If lights appear too weak in other apps, raise intensity in the Properties panel before export.
- For broader format support (FBX, Collada, etc.) the recommended path is: export GLB → convert with Blender or assimp / gltf-pipeline. Native multi-format export is planned.

## Rendering
- Viewport is real-time with optional shadows.
- **View > Render Image…** saves a PNG of the current 3D view (resolution configurable in the Render panel).
- **View > Render Animation…** renders the current timeline range to `animation.avi` (Motion-JPEG).
  The encoder is the pure-Rust `forge-video` crate: frames are JPEG-compressed on a pool of worker threads
  while the GPU renders the next frame, and streamed straight to disk. There is no C/C++ code, no `ffmpeg`
  and no extra build tools, so it builds and runs on every target the rest of the app supports.
  The AVI opens in VLC, mpv, QuickTime, Windows players and video editors (Blender, DaVinci, Premiere).
  Need an H.264 MP4? `ffmpeg -i animation.avi -c:v libx264 -crf 18 animation.mp4`.

## Recent files
File > Open Recent lists the last 10 projects you opened or saved. The list is stored in `recent.txt` in your config folder (APPDATA\BlendCraft on Windows, ~/Library/Application Support/BlendCraft on macOS, ~/.config/BlendCraft on Linux).

## Multi-platform builds (GitHub Actions)

Push to `main` / open a PR, or run **Actions → Build → Run workflow**.

| Target | Arch | Runner / method |
|--------|------|------------------|
| Linux | x64, x86, ARM64, ARMv7 | `ubuntu-latest` (+ `cross` where needed) |
| Windows | x64, x86, ARM64 | `windows-latest` |
| macOS | x64 (Intel), ARM64 (Apple Silicon) | `macos-13` / `macos-14` |
| FreeBSD | x64 | FreeBSD VM action |
| WebAssembly | wasm32 | `cargo check` for core/io; full GUI app needs web entry |

Artifacts appear under the workflow run. Tag a release as `v0.1.0` (etc.) to attach binaries to the GitHub Release.

### Local targets

```bash
rustup target add x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu \
  x86_64-pc-windows-msvc aarch64-apple-darwin wasm32-unknown-unknown

cargo build --release -p forge-app --target x86_64-unknown-linux-gnu
```

Cross-compiling to other OS/arch pairs is easiest with [cross](https://github.com/cross-rs/cross):

```bash
cargo install cross --git https://github.com/cross-rs/cross
cross build --release -p forge-app --target aarch64-unknown-linux-gnu
```

## Offline operation

BlendCraft's modeling, UV editing, project saving, and texture painting are local operations. The optional font catalog is shown as offline reference information only; fonts must be imported from files on the local computer. The application no longer includes the `ureq` HTTP client or a font-download action, and it does not upload project, mesh, UV, or texture data to a server.
