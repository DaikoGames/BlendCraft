# BlendCraft packaging

All recipes live under `packaging/`. Run what your OS supports:

```bash
# Linux host — tarball, deb, rpm, AppImage (Flatpak/Snap if tools installed)
bash packaging/build-all.sh

# Or one format:
bash packaging/appimage/build.sh
bash packaging/linux/build-deb-rpm.sh
bash packaging/linux/build-tarball.sh
bash packaging/flatpak/build.sh    # needs flatpak-builder
bash packaging/snap/build.sh       # needs snapcraft
bash packaging/macos/build-app-dmg.sh   # macOS only
bash packaging/web/build-zip.sh
bash packaging/bsd/build.sh             # FreeBSD host
```

Windows (PowerShell):

```powershell
.\packaging\windows\build-portable.ps1
# Inno Setup: open packaging\windows\BlendCraft.iss and compile
.\packaging\windows\build-msi.ps1
```

## Formats

| Format | Path | Output |
|--------|------|--------|
| **AppImage** | `packaging/appimage/` | `BlendCraft-x86_64.AppImage` |
| **Flatpak** | `packaging/flatpak/` | `BlendCraft.flatpak` |
| **Snap** | `packaging/snap/` | `.snap` |
| **deb** | `packaging/linux/nfpm.yaml` | `.deb` |
| **rpm** | `packaging/linux/nfpm.yaml` | `.rpm` |
| **Linux tarball** | `packaging/linux/build-tarball.sh` | `.tar.gz` |
| **Windows portable ZIP** | `packaging/windows/build-portable.ps1` | `.zip` |
| **Windows installer** | `packaging/windows/BlendCraft.iss` | Inno Setup `.exe` |
| **Windows MSI** | `packaging/windows/build-msi.ps1` | `.msi` via cargo-wix |
| **macOS .app + DMG** | `packaging/macos/build-app-dmg.sh` | `BlendCraft.app`, `.dmg` |
| **Web ZIP** | `packaging/web/build-zip.sh` | `BlendCraft-web.zip` |
| **BSD tarball / ports** | `packaging/bsd/` | `.tar.gz` + Makefile skeleton |

CI release packaging jobs produce the formats that GitHub runners can build (Linux tarball/deb/rpm/AppImage, Windows ZIP, macOS DMG, Web ZIP).
