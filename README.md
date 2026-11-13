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
it. The graph is stored in `.forge` projects, which are written as FRG4; existing FRG3 projects
remain readable.

This is a functional starter compositor, not full Blender compositor parity. It currently uses
8-bit RGBA CPU image operations and does not implement Blender's full node catalog, multi-input
compositing, masks, multilayer render passes, or color-management pipeline.

## Scripting
The **Scripting** workspace runs Python in an embedded CPython runtime from a background worker,
so scripts do not block the UI. BlendCraft itself and its scene bridge remain Rust; the small
`bpy`-named compatibility module is implemented natively in Rust and is not copied from Blender.
The project must be built with CPython development libraries available, and packaged builds need
the corresponding CPython runtime.

The implemented subset supports scene object lookup and selection, object names and transforms,
`mathutils.Vector`/`Euler`/`Quaternion`, common mesh-add operators, and basic material creation,
color, metallic/roughness, and single-slot object assignment through `bpy.data.materials` and
`obj.data.materials`. Scene changes are applied together as one undoable operation only after a
script finishes successfully. This is not full `bpy` compatibility: node trees, modifiers, mesh
editing, UI context, rendering, external Blender modules, and many other Blender systems are not
implemented. Scripts using unsupported APIs will fail explicitly rather than being silently
emulated. Use `examples/blendcraft_blender_compat_test.py` in either app.

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

## Not yet there
Vertex/edge/face edit mode, modifiers, UV editing, animation, video export, FBX / native .blend export.
Point-light omnidirectional shadows, cascaded shadow maps for large scenes.

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
