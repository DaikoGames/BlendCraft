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
