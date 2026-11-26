# Windows packaging

| Script / file | Output |
|---------------|--------|
| `build-portable.ps1` | `BlendCraft-windows-x64.zip` |
| `BlendCraft.iss` | Inno Setup installer (run with Inno Setup Compiler) |
| `build-msi.ps1` | MSI via `cargo-wix` |

MSIX is typically produced with Windows Packaging Project / `makeappx` after a Store-oriented build; CI can zip the portable layout for MSIX tooling offline.
