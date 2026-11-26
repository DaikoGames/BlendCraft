# MSI via WiX (requires WiX Toolset / cargo-wix)
# Install: cargo install cargo-wix
# Then from repo root: cargo wix -p forge-desktop
$ErrorActionPreference = "Stop"
Write-Host "Building MSI with cargo-wix (forge-desktop must define [[package.metadata.wix]] or use generated wxs)"
if (-not (Get-Command cargo-wix -ErrorAction SilentlyContinue)) {
  cargo install cargo-wix
}
Set-Location ..\..
cargo wix -p forge-desktop --nocapture
Write-Host "MSI should appear under target/wix/"
