//! Browser file download helper used only on `wasm32`.
//! Desktop continues to use `rfd` and the normal filesystem.
//!
//! Opening files from the browser is handled by the HTML file input /
//! drag-and-drop path; this module focuses on producing a downloadable
//! project or export without a backend.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::JsCast;
use web_sys::{Blob, BlobPropertyBag, Url};

/// Trigger a browser download of `data` under `filename`.
///
/// `mime` should be a standard MIME type, e.g. `"application/octet-stream"`
/// for `.forge` projects or `"model/gltf-binary"` for GLB.
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
