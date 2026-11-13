fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" || target_os == "freebsd" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/python/lib");
    } else if target_os == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/python/lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/python/Frameworks");
    }

    // Re-run when the logo assets change so the embedded icon stays in sync.
    println!("cargo:rerun-if-changed=icon.ico");
    println!("cargo:rerun-if-changed=logo.png");

    // Embed the application icon into the Windows executable so File Explorer,
    // the taskbar, Alt+Tab and the Start menu show the project logo.
    if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("icon.ico");
        res.set("ProductName", "BlendCraft");
        res.set("FileDescription", "BlendCraft – open-source 3D suite");
        res.set("OriginalFilename", "forge3d.exe");
        match res.compile() {
            Ok(()) => {}
            Err(e) => {
                // On native Windows builds this must succeed.
                // When cross-compiling from Linux without a resource compiler,
                // print a warning instead of failing the whole build.
                let target = std::env::var("TARGET").unwrap_or_default();
                let host = std::env::var("HOST").unwrap_or_default();
                if target.contains("windows") && host.contains("windows") {
                    panic!("winresource failed to embed icon.ico: {e}");
                } else {
                    eprintln!(
                        "cargo:warning=winresource could not embed icon (need a Windows RC on PATH): {e}"
                    );
                }
            }
        }
    }
}
