//! # forge-video
//!
//! Pure-Rust animation export for BlendCraft. No C/C++ code, no `cmake`, no
//! `nasm`, no `libclang`, no external `ffmpeg` binary: it compiles anywhere
//! `std` does, which is what makes it work on every OS / CPU target.
//!
//! ## Format
//! Motion-JPEG in an AVI container. Every frame is an independent JPEG, so
//! encoding is *embarrassingly parallel*: frames are compressed on a pool of
//! worker threads while the GPU is already rendering the next frame, then
//! written to disk in order. AVI/MJPEG opens in VLC, mpv, ffmpeg, QuickTime,
//! Windows media players and every video editor (Blender, DaVinci, Premiere).
//!
//! ## Pipeline
//! ```text
//! push_rgba() --> [bounded queue] --> N encoder threads --> writer thread --> file
//!                                        (JPEG per frame)    (re-orders frames,
//!                                                             streams to disk)
//! ```
//! Memory stays bounded (the queue is limited) and the file is streamed, so
//! long animations never have to fit in RAM.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use image::codecs::jpeg::JpegEncoder;
use image::{ExtendedColorType, ImageEncoder};

/// Classic (non-OpenDML) AVI uses 32-bit sizes; stay safely below 2 GiB.
const MAX_AVI_BYTES: u64 = 0x7FFF_0000;

/// What was written once [`VideoWriter::finish`] succeeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoStats {
    pub frames: u32,
    pub bytes: u64,
}

#[derive(Clone, Copy)]
struct Params {
    width: u32,
    height: u32,
    fps: u32,
}

type EncodeResult = (u32, io::Result<Vec<u8>>);

/// Streaming, multi-threaded MJPEG/AVI writer.
pub struct VideoWriter {
    width: u32,
    height: u32,
    next_index: u32,
    job_tx: Option<SyncSender<(u32, Vec<u8>)>>,
    workers: Vec<JoinHandle<()>>,
    writer: Option<JoinHandle<io::Result<VideoStats>>>,
}

impl VideoWriter {
    /// Create `path` and start the encoder threads.
    ///
    /// * `width`/`height` – frame size of the RGBA buffers you will push.
    /// * `fps` – playback frame rate (clamped to at least 1).
    /// * `quality` – JPEG quality 1..=100 (90 is visually near-lossless).
    pub fn create(
        path: impl AsRef<Path>,
        width: u32,
        height: u32,
        fps: u32,
        quality: u8,
    ) -> io::Result<Self> {
        if width == 0 || height == 0 || width > 65_535 || height > 65_535 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "video size must be 1..=65535 pixels in each dimension",
            ));
        }
        let params = Params {
            width,
            height,
            fps: fps.max(1),
        };
        let quality = quality.clamp(1, 100);
        let file = File::create(path.as_ref())?;
        let out = BufWriter::with_capacity(1 << 20, file);

        // Leave one core for the GPU/render thread, but always use at least one.
        let threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .saturating_sub(1)
            .clamp(1, 16);

        let (job_tx, job_rx) = mpsc::sync_channel::<(u32, Vec<u8>)>(threads * 2);
        let job_rx = Arc::new(Mutex::new(job_rx));
        let (res_tx, res_rx) = mpsc::channel::<EncodeResult>();

        let mut workers = Vec::with_capacity(threads);
        for _ in 0..threads {
            let job_rx = Arc::clone(&job_rx);
            let res_tx = res_tx.clone();
            workers.push(thread::spawn(move || loop {
                let job = {
                    let guard = job_rx.lock().unwrap_or_else(|e| e.into_inner());
                    guard.recv()
                };
                match job {
                    Ok((index, rgba)) => {
                        let encoded = encode_jpeg(&rgba, params.width, params.height, quality);
                        if res_tx.send((index, encoded)).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }));
        }
        // Only the workers hold senders now, so the writer ends when they all exit.
        drop(res_tx);

        let writer = thread::spawn(move || writer_loop(out, params, res_rx));

        Ok(Self {
            width,
            height,
            next_index: 0,
            job_tx: Some(job_tx),
            workers,
            writer: Some(writer),
        })
    }

    /// Queue one RGBA8 frame (`width * height * 4` bytes, alpha is ignored).
    /// Blocks only when the encoder queue is full, which keeps RAM bounded.
    pub fn push_rgba(&mut self, rgba: Vec<u8>) -> io::Result<()> {
        let needed = self.width as usize * self.height as usize * 4;
        if rgba.len() < needed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("frame buffer too small: {} < {needed} bytes", rgba.len()),
            ));
        }
        let tx = self
            .job_tx
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "video writer already finished"))?;
        tx.send((self.next_index, rgba)).map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "video encoder stopped (disk full or encode error); call finish() for details",
            )
        })?;
        self.next_index += 1;
        Ok(())
    }

    /// Wait for all queued frames, write the AVI index and finalize the file.
    pub fn finish(mut self) -> io::Result<VideoStats> {
        drop(self.job_tx.take());
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
        let writer = self
            .writer
            .take()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "video writer already finished"))?;
        writer
            .join()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "video writer thread panicked"))?
    }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

fn encode_jpeg(rgba: &[u8], width: u32, height: u32, quality: u8) -> io::Result<Vec<u8>> {
    let px = width as usize * height as usize;
    if rgba.len() < px * 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "frame buffer too small",
        ));
    }
    let mut rgb = vec![0u8; px * 3];
    for (dst, src) in rgb.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
        dst.copy_from_slice(&src[..3]);
    }
    let mut out: Vec<u8> = Vec::with_capacity(px / 4 + 1024);
    JpegEncoder::new_with_quality(&mut out, quality)
        .write_image(&rgb, width, height, ExtendedColorType::Rgb8)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// AVI container
// ---------------------------------------------------------------------------

fn put_u16(v: &mut Vec<u8>, x: u16) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn put_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

/// RIFF chunk: tag, size, data, pad byte to an even length.
fn chunk(tag: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(data.len() + 9);
    v.extend_from_slice(tag);
    put_u32(&mut v, data.len() as u32);
    v.extend_from_slice(data);
    if data.len() % 2 == 1 {
        v.push(0);
    }
    v
}

/// RIFF LIST: "LIST", size, list type, contents.
fn list(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(data.len() + 12);
    v.extend_from_slice(b"LIST");
    put_u32(&mut v, data.len() as u32 + 4);
    v.extend_from_slice(kind);
    v.extend_from_slice(data);
    v
}

/// Everything before the first frame chunk: `RIFF`, `hdrl`, and the `movi`
/// list header. Its length does not depend on the values, so it is written
/// once with placeholders and rewritten in place when the file is finished.
fn build_header(p: &Params, frames: u32, max_chunk: u32, movi_payload: u32) -> Vec<u8> {
    // avih: MainAVIHeader (56 bytes)
    let mut avih = Vec::with_capacity(56);
    put_u32(&mut avih, 1_000_000 / p.fps); // dwMicroSecPerFrame
    put_u32(&mut avih, max_chunk.saturating_mul(p.fps)); // dwMaxBytesPerSec
    put_u32(&mut avih, 0); // dwPaddingGranularity
    put_u32(&mut avih, 0x10 | 0x100); // AVIF_HASINDEX | AVIF_ISINTERLEAVED
    put_u32(&mut avih, frames); // dwTotalFrames
    put_u32(&mut avih, 0); // dwInitialFrames
    put_u32(&mut avih, 1); // dwStreams
    put_u32(&mut avih, max_chunk); // dwSuggestedBufferSize
    put_u32(&mut avih, p.width);
    put_u32(&mut avih, p.height);
    for _ in 0..4 {
        put_u32(&mut avih, 0); // dwReserved[4]
    }

    // strh: AVIStreamHeader (56 bytes)
    let mut strh = Vec::with_capacity(56);
    strh.extend_from_slice(b"vids");
    strh.extend_from_slice(b"MJPG");
    put_u32(&mut strh, 0); // dwFlags
    put_u16(&mut strh, 0); // wPriority
    put_u16(&mut strh, 0); // wLanguage
    put_u32(&mut strh, 0); // dwInitialFrames
    put_u32(&mut strh, 1); // dwScale
    put_u32(&mut strh, p.fps); // dwRate  (rate / scale = fps)
    put_u32(&mut strh, 0); // dwStart
    put_u32(&mut strh, frames); // dwLength
    put_u32(&mut strh, max_chunk); // dwSuggestedBufferSize
    put_u32(&mut strh, 0xFFFF_FFFF); // dwQuality (default)
    put_u32(&mut strh, 0); // dwSampleSize
    put_u16(&mut strh, 0); // rcFrame.left
    put_u16(&mut strh, 0); // rcFrame.top
    put_u16(&mut strh, p.width.min(65_535) as u16); // rcFrame.right
    put_u16(&mut strh, p.height.min(65_535) as u16); // rcFrame.bottom

    // strf: BITMAPINFOHEADER (40 bytes)
    let mut strf = Vec::with_capacity(40);
    put_u32(&mut strf, 40); // biSize
    put_u32(&mut strf, p.width); // biWidth
    put_u32(&mut strf, p.height); // biHeight
    put_u16(&mut strf, 1); // biPlanes
    put_u16(&mut strf, 24); // biBitCount
    strf.extend_from_slice(b"MJPG"); // biCompression
    put_u32(
        &mut strf,
        p.width.saturating_mul(p.height).saturating_mul(3),
    ); // biSizeImage
    put_u32(&mut strf, 0); // biXPelsPerMeter
    put_u32(&mut strf, 0); // biYPelsPerMeter
    put_u32(&mut strf, 0); // biClrUsed
    put_u32(&mut strf, 0); // biClrImportant

    let mut strl_body = chunk(b"strh", &strh);
    strl_body.extend_from_slice(&chunk(b"strf", &strf));
    let mut hdrl_body = chunk(b"avih", &avih);
    hdrl_body.extend_from_slice(&list(b"strl", &strl_body));
    let hdrl = list(b"hdrl", &hdrl_body);

    // RIFF size = "AVI " + hdrl + movi list (12 + payload) + idx1 chunk.
    let riff_size = 4u64 + hdrl.len() as u64 + 12 + movi_payload as u64 + 8 + 16 * frames as u64;

    let mut out = Vec::with_capacity(hdrl.len() + 24);
    out.extend_from_slice(b"RIFF");
    put_u32(&mut out, riff_size.min(u32::MAX as u64) as u32);
    out.extend_from_slice(b"AVI ");
    out.extend_from_slice(&hdrl);
    out.extend_from_slice(b"LIST");
    put_u32(&mut out, movi_payload.saturating_add(4));
    out.extend_from_slice(b"movi");
    out
}

fn writer_loop(
    mut out: BufWriter<File>,
    params: Params,
    rx: Receiver<EncodeResult>,
) -> io::Result<VideoStats> {
    let header_len = {
        let h = build_header(&params, 0, 0, 0);
        out.write_all(&h)?;
        h.len() as u64
    };

    let mut pending: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
    let mut next: u32 = 0;
    // (offset from the "movi" fourcc, size) per frame, for the idx1 index.
    let mut index: Vec<(u32, u32)> = Vec::new();
    let mut movi_off: u64 = 4;
    let mut max_chunk: u32 = 0;

    for (i, result) in rx {
        pending.insert(i, result?);
        while let Some(data) = pending.remove(&next) {
            let len = data.len() as u64;
            let padded = len + (len & 1);
            // movi_off counts the 4-byte "movi" tag, which build_header() already wrote.
            let projected =
                header_len + (movi_off - 4) + 8 + padded + 8 + 16 * (index.len() as u64 + 1);
            if projected > MAX_AVI_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "animation exceeds the 2 GiB AVI limit; lower the resolution or quality, or shorten the range",
                ));
            }
            out.write_all(b"00dc")?;
            out.write_all(&(len as u32).to_le_bytes())?;
            out.write_all(&data)?;
            if len & 1 == 1 {
                out.write_all(&[0])?;
            }
            index.push((movi_off as u32, len as u32));
            movi_off += 8 + padded;
            max_chunk = max_chunk.max(len as u32);
            next += 1;
        }
    }
    if !pending.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "video encoder lost a frame",
        ));
    }

    // idx1: one 16-byte entry per frame.
    let mut idx = Vec::with_capacity(index.len() * 16);
    for (off, size) in &index {
        idx.extend_from_slice(b"00dc");
        put_u32(&mut idx, 0x10); // AVIIF_KEYFRAME (every MJPEG frame is a key frame)
        put_u32(&mut idx, *off);
        put_u32(&mut idx, *size);
    }
    out.write_all(b"idx1")?;
    out.write_all(&(idx.len() as u32).to_le_bytes())?;
    out.write_all(&idx)?;

    let frames = index.len() as u32;
    let movi_payload = (movi_off - 4) as u32;
    let total = header_len + (movi_off - 4) + 8 + idx.len() as u64;

    // Rewrite the (fixed-length) header with the real frame count and sizes.
    out.seek(SeekFrom::Start(0))?;
    out.write_all(&build_header(&params, frames, max_chunk, movi_payload))?;
    out.flush()?;

    Ok(VideoStats {
        frames,
        bytes: total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rd32(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
    }

    fn synthetic(w: u32, h: u32, n: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                v.extend_from_slice(&[
                    (x * 255 / w.max(1)) as u8,
                    (y * 255 / h.max(1)) as u8,
                    (n * 40 % 256) as u8,
                    255,
                ]);
            }
        }
        v
    }

    fn roundtrip(w: u32, h: u32, frames: u32, fps: u32) {
        let path = std::env::temp_dir().join(format!(
            "forge_video_test_{}_{}x{}.avi",
            std::process::id(),
            w,
            h
        ));
        let mut vw = VideoWriter::create(&path, w, h, fps, 85).expect("create");
        for n in 0..frames {
            vw.push_rgba(synthetic(w, h, n)).expect("push");
        }
        let stats = vw.finish().expect("finish");
        assert_eq!(stats.frames, frames);

        let b = std::fs::read(&path).expect("read back");
        let _ = std::fs::remove_file(&path);
        assert_eq!(&b[0..4], b"RIFF");
        assert_eq!(rd32(&b, 4) as usize, b.len() - 8, "RIFF size");
        assert_eq!(&b[8..12], b"AVI ");
        assert_eq!(stats.bytes as usize, b.len());

        // hdrl -> avih: frame count + dimensions
        assert_eq!(&b[12..16], b"LIST");
        assert_eq!(&b[20..24], b"hdrl");
        assert_eq!(&b[24..28], b"avih");
        assert_eq!(rd32(&b, 32), 1_000_000 / fps, "microsec per frame");
        assert_eq!(rd32(&b, 48), frames, "dwTotalFrames");
        assert_eq!(rd32(&b, 64), w);
        assert_eq!(rd32(&b, 68), h);

        // movi list, then the first frame must be a JPEG (SOI marker).
        let movi = b
            .windows(4)
            .position(|x| x == b"movi")
            .expect("movi list present");
        assert_eq!(&b[movi + 4..movi + 8], b"00dc");
        assert_eq!(&b[movi + 12..movi + 14], &[0xFFu8, 0xD8u8]);

        // idx1 holds one entry per frame.
        let idx = b
            .windows(4)
            .rposition(|x| x == b"idx1")
            .expect("idx1 present");
        assert_eq!(rd32(&b, idx + 4), frames * 16);
    }

    #[test]
    fn writes_valid_avi() {
        roundtrip(64, 48, 6, 24);
    }

    #[test]
    fn handles_odd_sizes() {
        roundtrip(33, 17, 3, 30);
    }

    #[test]
    fn rejects_short_buffer() {
        let path =
            std::env::temp_dir().join(format!("forge_video_short_{}.avi", std::process::id()));
        let mut vw = VideoWriter::create(&path, 16, 16, 24, 90).unwrap();
        assert!(vw.push_rgba(vec![0; 10]).is_err());
        let _ = vw.finish();
        let _ = std::fs::remove_file(&path);
    }
}
