# Changes: Cycles engine + cloth

Please run:

    cargo test -p forge-core -p forge-render
    cargo build --release

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
