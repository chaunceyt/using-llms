// main.js — app bootstrap. Creates the renderer/scene/camera, binds the world,
// loads modules in dependency-safe order, and drives the frame loop.
import * as THREE from 'three';
import { createWorld, startModules, updateModules, clock, world, setTimeOfDay, setWeather } from './core/index.js';

// ---------------------------------------------------------------------------
// Module load order. Each wave depends only on waves before it.
// Wave 1: foundational systems + UI shell. Wave 2: city content. Wave 3: demo.
// ---------------------------------------------------------------------------
const WAVE_ORDER = [
  // wave 1
  'terrain', 'environment', 'roads', 'simulation', 'effects',
  // wave 2 (content — needs terrain/roads/simulation)
  'zoning', 'buildings', 'props', 'traffic', 'tools',
  // ui last-ish so it can render status from other modules; audio near end
  'audio', 'ui',
];

// demo is opt-in via ?demo=1 (or always after the core city exists).
const wantDemo = new URLSearchParams(location.search).has('demo');
if (wantDemo) WAVE_ORDER.push('demo');

// ---------------------------------------------------------------------------
// Renderer with graceful software fallback (helps headless screenshots).
// ---------------------------------------------------------------------------
function createRenderer() {
  // three needs a real <canvas>; #app is the layout container, so append one.
  const app = document.getElementById('app');
  const canvas = document.createElement('canvas');
  if (app) app.appendChild(canvas);
  try {
    const r = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: 'high-performance',
      stencil: false,
      // lets the screenshot tool read pixels back via toDataURL any time
      preserveDrawingBuffer: true,
    });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    return r;
  } catch (e) {
    console.warn('[main] WebGL unavailable, trying WebGL1 fallback', e);
    const r = new THREE.WebGLRenderer({ canvas, antialias: true });
    r.setPixelRatio(1);
    return r;
  }
}

const renderer = createRenderer();
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

// ---------------------------------------------------------------------------
// Scene + camera (world units in metres, +Y up). Camera will be reparented by
// the demo/tools modules for flyovers; we just give it a sane default.
// ---------------------------------------------------------------------------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a1117);

const camera = new THREE.PerspectiveCamera(
  55, window.innerWidth / window.innerHeight, 0.1, 8000,
);
camera.position.set(320, 260, 420); // overview of a city chunk
camera.lookAt(0, 0, 0);

// Bind the world to this renderer/scene/camera.
createWorld({ seed: 1337, renderer, scene, camera });

// Handle resize.
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
renderer.setSize(window.innerWidth, window.innerHeight);

// ---------------------------------------------------------------------------
// Boot modules. If a module fails to load we surface it in the UI but keep going.
// ---------------------------------------------------------------------------
const boot = startModules(WAVE_ORDER);

async function ready() {
  await boot;
  console.log('[main] modules loaded:', Object.keys(world.modules).join(', '));
  document.body.dataset.ready = '1';
  window.__skylinesReady = true;
  renderer.render(scene, camera);
  // Hooks for the screenshot tool (see tools/screenshot.mjs).
  window.__skylines = {
    scene, camera, renderer, world,
    setTimeOfDay, setWeather,
    getStats: () => ({ fps: world.stats.fps, drawCalls: world.stats.drawCalls }),
  };
  window.__grabFrame = () => {
    try { return renderer.domElement.toDataURL('image/png'); }
    catch (e) { console.error('[main] grabFrame failed', e); return null; }
  };
  window.__skylines.setCamera = (pos, target) => {
    camera.position.fromArray(pos);
    if (target) camera.lookAt(...target);
    else camera.updateMatrixWorld();
  };
  // Let the screenshot tool know we are presentable.
  window.dispatchEvent(new CustomEvent('skylines:ready'));
}
ready();

// ---------------------------------------------------------------------------
// Frame loop: advance the fixed-step simulation clock, run module updates,
// then render. Module failures are isolated by updateModules().
// ---------------------------------------------------------------------------
renderer.setAnimationLoop(() => {
  clock.tick(performance.now());
  updateModules();
  renderer.render(scene, camera);
  world.stats.drawCalls = renderer.info.render.calls;
  world.stats.frameTimeMs = (renderer.info.render.calls > 0 ? performance.now() : 0);
});

export { scene, camera, renderer };
