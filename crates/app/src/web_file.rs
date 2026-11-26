//! Browser-native file pick + download helpers (`wasm32` only).
//!
//! Uses the **browser’s file picker** (`<input type="file">`), which on each OS
//! surfaces that OS’s picker inside Chrome/Firefox/Safari. Desktop builds keep
//! using `rfd` (Win32 / GTK / NSOpenPanel) via the non-wasm paths in `lib.rs`.

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{Blob, BlobPropertyBag, FileReader, HtmlInputElement, Url};

/// What the user just picked (filled asynchronously by the file input).
#[derive(Clone)]
pub struct PickedFile {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// Shared slot written by the browser file-input callback and read by the app each frame.
pub type PendingPick = Rc<RefCell<Option<PickedFile>>>;

pub fn new_pending() -> PendingPick {
    Rc::new(RefCell::new(None))
}

/// Trigger the browser/OS file picker. `accept` is an HTML accept string, e.g.
/// `".forge,.fbx,.glb,.obj,application/octet-stream"`.
pub fn pick_file(accept: &str, pending: PendingPick) {
    let Some(window) = web_sys::window() else { return };
    let Some(document) = window.document() else { return };
    let Ok(input) = document.create_element("input") else { return };
    let Ok(input) = input.dyn_into::<HtmlInputElement>() else { return };
    input.set_type("file");
    input.set_accept(accept);
    // Keep element alive until change fires.
    let input_clone = input.clone();
    let onchange = Closure::once_into_js(move |_event: web_sys::Event| {
        let files = match input_clone.files() {
            Some(f) if f.length() > 0 => f,
            _ => return,
        };
        let Some(file) = files.item(0) else { return };
        let name = file.name();
        let reader = match FileReader::new() {
            Ok(r) => r,
            Err(_) => return,
        };
        let reader_c = reader.clone();
        let pending_c = pending.clone();
        let onload = Closure::once_into_js(move |_e: web_sys::Event| {
            let Ok(result) = reader_c.result() else { return };
            let Ok(array_buf) = result.dyn_into::<js_sys::ArrayBuffer>() else { return };
            let arr = js_sys::Uint8Array::new(&array_buf);
            let mut bytes = vec![0u8; arr.length() as usize];
            arr.copy_to(&mut bytes);
            *pending_c.borrow_mut() = Some(PickedFile { name, bytes });
        });
        reader.set_onload(Some(onload.as_ref().unchecked_ref()));
        // Leak the onload closure for the lifetime of the read (once fires once).
        std::mem::forget(onload);
        let _ = reader.read_as_array_buffer(&file);
    });
    input.set_onchange(Some(onchange.as_ref().unchecked_ref()));
    std::mem::forget(onchange);
    input.click();
}

/// Trigger a browser download of `data` under `filename`.
pub fn download_bytes(filename: &str, data: &[u8], mime: &str) {
    let Some(window) = web_sys::window() else { return };
    let Some(document) = window.document() else { return };

    let array = js_sys::Uint8Array::new_with_length(data.len() as u32);
    array.copy_from(data);

    let mut props = BlobPropertyBag::new();
    props.type_(mime);
    let parts = js_sys::Array::new();
    parts.push(&array);
    let Ok(blob) = Blob::new_with_u8_array_sequence_and_options(&parts, &props) else {
        return;
    };
    let Ok(url) = Url::create_object_url_with_blob(&blob) else {
        return;
    };

    if let Ok(a) = document.create_element("a") {
        if let Ok(a) = a.dyn_into::<web_sys::HtmlElement>() {
            let _ = a.set_attribute("href", &url);
            let _ = a.set_attribute("download", filename);
            a.click();
        }
    }
    let _ = Url::revoke_object_url(&url);
}

/// Encode RGBA8 to PNG bytes for still-image download.
pub fn encode_png(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let mut png = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut png);
    use image::ImageEncoder;
    encoder
        .write_image(rgba, width, height, image::ExtendedColorType::Rgba8)
        .map_err(|e| e.to_string())?;
    Ok(png)
}


/// Encode RGBA frames to a single video via browser APIs (prefers MP4).
/// Uses `globalThis.__bc_encodeVideo` installed by `bootstrap.js`.
/// Returns (filename, mime, bytes).
pub async fn encode_animation_video(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    fps: u16,
) -> Result<(String, String, Vec<u8>), String> {
    use js_sys::{Array, Reflect, Uint8Array};
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let global = js_sys::global();
    let encode = Reflect::get(&global, &"__bc_encodeVideo".into())
        .map_err(|_| "__bc_encodeVideo missing — encode_video.js failed to load".to_string())?;
    if encode.is_undefined() || encode.is_null() {
        return Err("__bc_encodeVideo missing — encode_video.js failed to load".into());
    }
    let encode_fn = encode
        .dyn_into::<js_sys::Function>()
        .map_err(|_| "__bc_encodeVideo is not a function".to_string())?;

    let frames_js = Array::new();
    for f in frames {
        frames_js.push(&Uint8Array::from(f.as_slice()));
    }

    let call = encode_fn
        .call4(
            &JsValue::UNDEFINED,
            &frames_js,
            &JsValue::from(width),
            &JsValue::from(height),
            &JsValue::from(fps as u32),
        )
        .map_err(|e| format!("encodeVideo call failed: {e:?}"))?;
    let result = JsFuture::from(js_sys::Promise::resolve(&call))
        .await
        .map_err(|e| format!("encodeVideo rejected: {e:?}"))?;

    let filename = Reflect::get(&result, &"filename".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "animation.mp4".into());
    let mime = Reflect::get(&result, &"mime".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "video/mp4".into());
    let data_val = Reflect::get(&result, &"data".into()).map_err(|e| format!("{e:?}"))?;
    let data = data_val
        .dyn_into::<Uint8Array>()
        .map_err(|_| "encodeVideo data is not Uint8Array".to_string())?;
    let mut bytes = vec![0u8; data.length() as usize];
    data.copy_to(&mut bytes);
    Ok((filename, mime, bytes))
}
