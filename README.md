# Forge3D (working title)

this is a repository to test out the abilities i can have with Claude - i am trying to remake Blender in Rust from the ground up. 
Huge Thanks to the Blender Foundation making the Blender Project, have a visit at their site too: https://www.blender.org/

Rust 3D suite for game assets.

- Step 1: workspace, viewport, glTF viewer
- Step 2: PBR materials/textures, HDRI lighting, view modes
- Step 3 (this): egui UI (menu bar, outliner, properties), parametric primitives, selection,
  transforms, materials, undo/redo, project save/load (.forge), import glTF/GLB/OBJ, export GLB/OBJ

## Run
```
cargo run --release -p forge-app
cargo run --release -p forge-app -- model.glb
cargo test -p forge-core
```

## Workflow
1. Add > Mesh > Cube / Sphere / ... then tweak parameters, transform and material in the Properties panel.
2. Combine objects to build a model. Click objects in the viewport to select them.
3. File > Save (.forge) to keep the project, File > Export > glTF Binary to use it in Unity/Godot.

## Not yet there
Vertex/edge/face edit mode, modifiers, UV editing, animation, lights/cameras, gizmos.

## Gizmo
Select an object, press G (move), R (rotate) or S (scale) or use the toolbar in the viewport, then drag the colored handles. Q returns to select mode. Every drag is one undo step.

## Object types (Add menu)
- Mesh: Plane, Cube, Circle, UV Sphere, Ico Sphere, Cylinder, Cone, Torus, Grid, Capsule (parameters stay editable)
- Empty: Plain Axes, Arrows, Single Arrow, Circle, Cube, Sphere, Cone
- Light: Point, Sun, Spot (real-time, no shadows yet; up to 8 lights)
- Camera: Perspective / Orthographic. Press 0 (or View > Camera view) to look through it.
- Image as Plane: textured upright quad from a PNG/JPG
- GLB export includes cameras, lights (KHR_lights_punctual) and empties.
- Greyed out in the menu (need bigger systems first): Curves, Surface, Metaball, Text, Volume, Grease Pencil,
  Armature, Lattice, Area light, Light Probe, Speaker, Force Field, Collection Instance.

Note: project files (.forge) from the previous step cannot be opened (format changed to FRG2).

## Object mode (complete set)
- Selection: click, Shift+click, Box / Lasso / Circle tools, Select All / None / Invert / Random,
  Select Children / Parent / Similar (type, collection, material, linked).
- Transform: gizmo (Move / Rotate / Scale) and modal G / R / S with axis constraints, plane constraints,
  numeric input, snapping (increment or vertex), pivot points, transform orientations (Global / Local / View),
  3D cursor (Shift+Right click).
- Duplicate (Shift+D), linked duplicate (Alt+D), single user, delete, hide / show.
- Parenting (Ctrl+P), collections (flat), outliner tree with filter, rename (F2 / double click).
- Set origin, geometry to origin, apply / clear transforms, align objects, snap selection / cursor.
- Join (Ctrl+J, mixed materials become slots) and Separate (by loose parts, by material).
- Constraints: Copy Location / Rotation / Scale / Transforms, Track To, Limit Location / Rotation / Scale,
  Limit Distance, Floor.
- Help > Keyboard Shortcuts (or F1) lists every live shortcut and is generated from the key table.

Note: .forge projects from earlier steps cannot be opened (format FRG3).

## Export
File > Export has glTF Binary and Wavefront OBJ, each also as "selected only".

## Recent files
File > Open Recent lists the last 10 projects you opened or saved. The list is stored in `recent.txt` in your config folder (APPDATA\Forge3D on Windows, ~/Library/Application Support/Forge3D on macOS, ~/.config/Forge3D on Linux).
