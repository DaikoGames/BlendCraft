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
- **View > Render Animation…** renders the current timeline range as a PNG sequence plus an H.264 MP4.
  The MP4 is produced with the OpenH264 encoder (Cisco, real-time capable) and the pure-Rust minimp4 muxer —
  no system `ffmpeg` binary is required. This works out of the box on Windows, Linux, macOS and FreeBSD.

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
