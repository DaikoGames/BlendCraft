# Portable ZIP of BlendCraft for Windows
param(
  [string]$OutDir = "..\..\dist\packaging",
  [string]$Bin = "..\..\target\x86_64-pc-windows-msvc\release\forge3d.exe"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stage = Join-Path $OutDir "BlendCraft-windows-x64"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
if (-not (Test-Path $Bin)) {
  cargo build --release -p forge-desktop --target x86_64-pc-windows-msvc
}
Copy-Item $Bin (Join-Path $stage "forge3d.exe")
Copy-Item ..\..\LICENSE.md $stage -ErrorAction SilentlyContinue
Copy-Item ..\..\README.md $stage -ErrorAction SilentlyContinue
$zip = Join-Path $OutDir "BlendCraft-windows-x64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $stage -DestinationPath $zip
Write-Host "Portable ZIP: $zip"
