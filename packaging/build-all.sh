#!/usr/bin/env bash
# Build every packaging format that this host can produce.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/dist/packaging}"
mkdir -p "$OUT"
OS="$(uname -s)"

echo "==> Ensuring release binary"
case "$OS" in
  Linux)
    cargo build --release -p forge-desktop --target x86_64-unknown-linux-gnu || cargo build --release -p forge-desktop
    bash "$ROOT/packaging/linux/build-tarball.sh" "$OUT"
    bash "$ROOT/packaging/linux/build-deb-rpm.sh" "$OUT" || echo "nfpm deb/rpm skipped"
    bash "$ROOT/packaging/appimage/build.sh" "$OUT" || echo "AppImage skipped"
    if command -v flatpak-builder >/dev/null 2>&1; then
      bash "$ROOT/packaging/flatpak/build.sh" "$OUT" || echo "Flatpak skipped"
    else
      echo "flatpak-builder not installed — skip Flatpak"
    fi
    if command -v snapcraft >/dev/null 2>&1; then
      bash "$ROOT/packaging/snap/build.sh" "$OUT" || echo "Snap skipped"
    else
      echo "snapcraft not installed — skip Snap"
    fi
    ;;
  Darwin)
    cargo build --release -p forge-desktop
    bash "$ROOT/packaging/macos/build-app-dmg.sh" "$OUT"
    ;;
  MINGW*|MSYS*|CYGWIN*|Windows_NT)
    echo "On Windows run packaging/windows/build-portable.ps1 and compile BlendCraft.iss"
    ;;
  *)
    cargo build --release -p forge-desktop
    bash "$ROOT/packaging/bsd/build.sh" "$OUT" || true
    ;;
esac

if [[ -f "$ROOT/web/build.sh" ]]; then
  bash "$ROOT/packaging/web/build-zip.sh" "$OUT" || echo "web zip skipped"
fi

echo "Done. Artifacts in $OUT"
ls -la "$OUT" || true
