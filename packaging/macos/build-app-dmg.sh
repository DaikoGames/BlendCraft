#!/usr/bin/env bash
# Create BlendCraft.app and a DMG (run on macOS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/dist/packaging}"
mkdir -p "$OUT"
APP="$OUT/BlendCraft.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

BIN=""
for c in \
  "$ROOT/dist/forge3d-universal-apple-darwin" \
  "$ROOT/target/aarch64-apple-darwin/release/forge3d" \
  "$ROOT/target/x86_64-apple-darwin/release/forge3d" \
  "$ROOT/target/release/forge3d"; do
  [[ -x "$c" ]] && BIN="$c" && break
done
if [[ -z "$BIN" ]]; then
  cargo build --release -p forge-desktop
  BIN="$ROOT/target/release/forge3d"
fi
cp "$BIN" "$APP/Contents/MacOS/forge3d"
chmod +x "$APP/Contents/MacOS/forge3d"

cat > "$APP/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>BlendCraft</string>
  <key>CFBundleDisplayName</key><string>BlendCraft</string>
  <key>CFBundleIdentifier</key><string>games.daiko.BlendCraft</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleExecutable</key><string>forge3d</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

if [[ -f "$ROOT/web/favicon.png" ]]; then
  cp "$ROOT/web/favicon.png" "$APP/Contents/Resources/AppIcon.png"
fi

DMG="$OUT/BlendCraft-macOS.dmg"
rm -f "$DMG"
hdiutil create -volname BlendCraft -srcfolder "$APP" -ov -format UDZO "$DMG"
echo "App: $APP"
echo "DMG: $DMG"
