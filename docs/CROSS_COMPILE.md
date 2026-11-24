# Building BlendCraft for RISC-V & FreeBSD ARM64

Goal: **run the full desktop app (with RustPython scripting)** on these platforms.

Web/WASM is optional and separate.

## Preferred: build on the device (or same OS/arch VM)

### Linux RISC-V (gnu)

```bash
# on a riscv64 Linux machine (or chroot/qemu user)
sudo apt-get install -y build-essential pkg-config \
  libx11-dev libxcursor-dev libxrandr-dev libxi-dev \
  libxkbcommon-dev libwayland-dev libgl1-mesa-dev \
  libasound2-dev libudev-dev
cargo build --release -p forge-app
./target/release/forge3d
```

Scripting is on by default (`python-scripting`).

### FreeBSD aarch64

```bash
# on FreeBSD/arm64
pkg install -y rust pkgconf
# X11 / graphics deps as needed for your setup
cargo build --release -p forge-app
./target/release/forge3d
```

## Cross-compile from x86_64 Linux

### RISC-V gnu (with scripting / SQLite)

```bash
sudo apt-get install -y gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
rustup target add riscv64gc-unknown-linux-gnu

export CC=riscv64-linux-gnu-gcc
export CXX=riscv64-linux-gnu-g++
export CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=riscv64-linux-gnu-gcc

cargo build --release -p forge-app --target riscv64gc-unknown-linux-gnu
# binary: target/riscv64gc-unknown-linux-gnu/release/forge3d
```

Copy the binary to a RISC-V Linux machine and run it there (matching glibc version helps).

### FreeBSD aarch64 (sysroot)

```bash
rustup target add aarch64-unknown-freebsd
REL=14.2-RELEASE
mkdir -p sysroot
curl -fsSL -o base.txz \
  "https://download.freebsd.org/releases/arm64/aarch64/${REL}/base.txz"
tar -xJf base.txz -C sysroot ./lib ./usr/lib ./usr/include ./libexec
```

See the `freebsd-arm64` job in `.github/workflows/rust.yml` for the clang `--sysroot` linker wrapper.

Native FreeBSD/arm64 is still the reliable path for day-to-day coding.

## CI

- **linux-riscv64** uses Ubuntu’s `gcc-riscv64-linux-gnu` (not Zig) so RustPython/SQLite can build.
- **freebsd-arm64** downloads FreeBSD `base.txz` and links with clang `--sysroot`.
- Both are experimental / continue-on-error until fully stable on GHA.

Zig is only used for experimental **i686** and **armv7** Linux cross builds.
