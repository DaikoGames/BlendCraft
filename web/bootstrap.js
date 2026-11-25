import init from './forge_web.js';

const loading = document.getElementById('loading');
try {
  await init();
  loading?.remove();
} catch (error) {
  if (loading) {
    loading.textContent = 'BlendCraft Web failed to start. Open the browser developer console for details.';
    loading.style.color = '#ff8c8c';
  }
  console.error('BlendCraft Web startup error:', error);
}
