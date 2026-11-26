/**
 * Encode RGBA frames to a single video file.
 * Prefers real MP4 (WebCodecs + mp4-muxer from CDN). Falls back to MediaRecorder
 * (MP4 on Safari, WebM on Chrome/Firefox if muxer unavailable).
 *
 * @param {Uint8Array[]} frames RGBA buffers
 * @param {number} width
 * @param {number} height
 * @param {number} fps
 * @returns {Promise<{ data: Uint8Array, mime: string, filename: string }>}
 */
export async function encodeVideo(frames, width, height, fps) {
  if (!frames || !frames.length) throw new Error('No frames to encode');
  fps = Math.max(1, fps | 0);
  width = width | 0;
  height = height | 0;

  // --- Preferred: WebCodecs H.264 → MP4 ---
  if (typeof VideoEncoder !== 'undefined') {
    try {
      const result = await encodeMp4WebCodecs(frames, width, height, fps);
      if (result) return result;
    } catch (e) {
      console.warn('BlendCraft: WebCodecs MP4 encode failed, trying MediaRecorder', e);
    }
  }

  // --- Fallback: MediaRecorder (mp4 if browser supports it, else webm) ---
  return encodeMediaRecorder(frames, width, height, fps);
}

async function encodeMp4WebCodecs(frames, width, height, fps) {
  let Muxer, ArrayBufferTarget;
  try {
    const mod = await import('https://cdn.jsdelivr.net/npm/mp4-muxer@4.3.3/+esm');
    Muxer = mod.Muxer;
    ArrayBufferTarget = mod.ArrayBufferTarget;
  } catch (e) {
    console.warn('BlendCraft: could not load mp4-muxer', e);
    return null;
  }

  const target = new ArrayBufferTarget();
  const muxer = new Muxer({
    target,
    video: {
      codec: 'avc',
      width,
      height,
    },
    fastStart: 'in-memory',
    firstTimestampBehavior: 'offset',
  });

  const config = {
    codec: 'avc1.42001f', // baseline
    width,
    height,
    bitrate: Math.max(1_000_000, width * height * fps * 0.25),
    framerate: fps,
    avc: { format: 'avc' },
  };

  const support = await VideoEncoder.isConfigSupported(config);
  if (!support.supported) {
    // try a more widely supported level
    config.codec = 'avc1.42E01E';
    const s2 = await VideoEncoder.isConfigSupported(config);
    if (!s2.supported) return null;
  }

  let encodeError = null;
  const encoder = new VideoEncoder({
    output: (chunk, meta) => {
      muxer.addVideoChunk(chunk, meta);
    },
    error: (e) => {
      encodeError = e;
    },
  });
  encoder.configure(config);

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  for (let i = 0; i < frames.length; i++) {
    const rgba = frames[i];
    const imageData = new ImageData(new Uint8ClampedArray(rgba.buffer, rgba.byteOffset, width * height * 4), width, height);
    ctx.putImageData(imageData, 0, 0);
    const bitmap = await createImageBitmap(canvas);
    const frame = new VideoFrame(bitmap, {
      timestamp: Math.round((i * 1_000_000) / fps),
      duration: Math.round(1_000_000 / fps),
    });
    encoder.encode(frame, { keyFrame: i % Math.max(1, fps) === 0 });
    frame.close();
    bitmap.close();
    if (encodeError) throw encodeError;
  }

  await encoder.flush();
  encoder.close();
  muxer.finalize();

  const data = new Uint8Array(target.buffer);
  return { data, mime: 'video/mp4', filename: 'animation.mp4' };
}

function encodeMediaRecorder(frames, width, height, fps) {
  return new Promise(async (resolve, reject) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      reject(new Error('2D canvas unavailable'));
      return;
    }

    const stream = canvas.captureStream(fps);
    const candidates = [
      'video/mp4;codecs=avc1.42E01E',
      'video/mp4;codecs=avc1',
      'video/mp4',
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
    ];
    let mime = '';
    for (const c of candidates) {
      if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(c)) {
        mime = c;
        break;
      }
    }
    if (!mime) {
      reject(new Error('No MediaRecorder video MIME type supported'));
      return;
    }

    const chunks = [];
    let recorder;
    try {
      recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8_000_000 });
    } catch (e) {
      reject(e);
      return;
    }

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size) chunks.push(e.data);
    };
    recorder.onerror = (e) => reject(e.error || new Error('MediaRecorder error'));
    recorder.onstop = async () => {
      try {
        const blob = new Blob(chunks, { type: mime });
        const buf = new Uint8Array(await blob.arrayBuffer());
        const isMp4 = mime.startsWith('video/mp4');
        resolve({
          data: buf,
          mime: isMp4 ? 'video/mp4' : 'video/webm',
          filename: isMp4 ? 'animation.mp4' : 'animation.webm',
        });
      } catch (e) {
        reject(e);
      }
    };

    recorder.start();
    const frameDelay = 1000 / fps;
    for (let i = 0; i < frames.length; i++) {
      const rgba = frames[i];
      const imageData = new ImageData(
        new Uint8ClampedArray(rgba.buffer, rgba.byteOffset, width * height * 4),
        width,
        height,
      );
      ctx.putImageData(imageData, 0, 0);
      // Let the capture stream sample this frame
      await new Promise((r) => setTimeout(r, frameDelay));
    }
    // One extra beat so the last frame is captured
    await new Promise((r) => setTimeout(r, frameDelay));
    recorder.stop();
    stream.getTracks().forEach((t) => t.stop());
  });
}
