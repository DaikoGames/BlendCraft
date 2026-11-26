#!/usr/bin/env bash
# Build a BlendCraft AppImage from a linux x86_64 release binary.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/dist/packaging}"
BIN="${BIN:-forge3d}"
ARCH="${ARCH:-x86_64}"
mkdir -p "$OUT"
APPDIR="$OUT/BlendCraft.AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"

# Prefer CI artifact, else build
if [[ -x "$ROOT/target/release/$BIN" ]]; then
  cp "$ROOT/target/release/$BIN" "$APPDIR/usr/bin/forge3d"
elif [[ -x "$ROOT/target/x86_64-unknown-linux-gnu/release/$BIN" ]]; then
  cp "$ROOT/target/x86_64-unknown-linux-gnu/release/$BIN" "$APPDIR/usr/bin/forge3d"
else
  cargo build --release -p forge-desktop --target x86_64-unknown-linux-gnu
  cp "$ROOT/target/x86_64-unknown-linux-gnu/release/$BIN" "$APPDIR/usr/bin/forge3d"
fi
chmod +x "$APPDIR/usr/bin/forge3d"

cp "$ROOT/packaging/appimage/BlendCraft.desktop" "$APPDIR/usr/share/applications/"
cp "$ROOT/packaging/appimage/BlendCraft.desktop" "$APPDIR/"
# Icon: reuse web favicon if present
if [[ -f "$ROOT/web/favicon.png" ]]; then
  cp "$ROOT/web/favicon.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/blendcraft.png"
  cp "$ROOT/web/favicon.png" "$APPDIR/blendcraft.png"
fi
ln -sf usr/bin/forge3d "$APPDIR/AppRun"

# Fetch appimagetool if needed
TOOL="$OUT/appimagetool-$ARCH.AppImage"
if [[ ! -x "$TOOL" ]]; then
  curl -fsSL -o "$TOOL" \
    "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${ARCH}.AppImage"
  chmod +x "$TOOL"
fi
(
  cd "$OUT"
  ARCH="$ARCH" "$TOOL" BlendCraft.AppDir "BlendCraft-${ARCH}.AppImage"
)
echo "AppImage: $OUT/BlendCraft-${ARCH}.AppImage"
