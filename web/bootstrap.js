// On-page log so problems are visible without opening DevTools.
const panel = document.createElement('pre');
panel.id = 'bc-log';
panel.style.cssText = 'position:fixed;left:0;right:0;bottom:0;max-height:40vh;overflow:auto;margin:0;padding:8px 12px;' +
  'font:12px/1.4 ui-monospace,monospace;background:rgba(10,10,12,.92);color:#cfd4dc;z-index:2147483647;white-space:pre-wrap;';
document.body.appendChild(panel);
const add = (kind, args) => {
  const text = args.map((a) => (a && a.stack) ? a.stack : (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' ');
  const line = document.createElement('div');
  line.textContent = `[${kind}] ${text}`;
  if (kind === 'error') line.style.color = '#ff8c8c';
  if (kind === 'warn') line.style.color = '#ffd27f';
  panel.appendChild(line);
  panel.scrollTop = panel.scrollHeight;
};
for (const kind of ['log', 'info', 'warn', 'error']) {
  const orig = console[kind].bind(console);
  console[kind] = (...args) => { orig(...args); add(kind, args); };
}
window.addEventListener('error', (e) => add('error', [e.message || e.error]));
window.addEventListener('unhandledrejection', (e) => add('error', ['Unhandled promise rejection:', e.reason]));

add('info', ['secure context:', window.isSecureContext, '| navigator.gpu:', 'gpu' in navigator, '| url:', location.href]);

const loading = document.getElementById('loading');
const fail = (msg) => {
  if (loading) { loading.textContent = msg; loading.style.color = '#ff8c8c'; }
  add('error', [msg]);
};

// Decide the graphics backend before the wasm module starts:
// WebGPU if a real adapter is available, otherwise WebGL2 (Firefox, older/blocked GPUs).
let webgpu = false;
try {
  if ('gpu' in navigator) {
    const adapter = await navigator.gpu.requestAdapter();
    webgpu = !!adapter;
  }
} catch (e) {
  add('warn', ['WebGPU probe failed:', e]);
}
globalThis.__bc_webgpu = webgpu;
let webgl2 = false;
try { webgl2 = !!document.createElement('canvas').getContext('webgl2'); } catch (_) {}
add('info', ['WebGPU adapter:', webgpu, '| WebGL2:', webgl2]);

if (!webgpu && !webgl2) {
  fail('Neither WebGPU nor WebGL2 is available. Enable hardware acceleration in your browser settings and update your graphics driver.');
} else {
  try {
    const { default: init } = await import('./forge_web.js');
    await init();
    add('info', ['wasm module started']);
    setTimeout(() => loading?.remove(), 1500);
  } catch (error) {
    fail('BlendCraft Web failed to start: ' + (error && error.message ? error.message : error));
    console.error('BlendCraft Web startup error:', error);
  }
}
