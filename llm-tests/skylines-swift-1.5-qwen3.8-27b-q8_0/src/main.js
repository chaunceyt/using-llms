// Core bootstrap (integrator-owned). Wires world + ctx + modules, runs the loop,
// and exposes the app handle used by the screenshot tool and the UI.
import * as THREE from 'three';
import { createWorld } from './core/world.js';
import { makeRNG } from './core/rng.js';
import { EventBus } from './core/event.js';
import { Clock } from './core/clock.js';
import { Perf } from './core/perf.js';
import { Assets } from './core/assets.js';
import { Registry } from './core/registry.js';
import { Loop } from './core/loop.js';

import Terrain from './terrain/index.js';
import Environment from './environment/index.js';
import Roads from './roads/index.js';
import Zoning from './zoning/index.js';
import Buildings from './buildings/index.js';
import Props from './props/index.js';
import Traffic from './traffic/index.js';
import Simulation from './simulation/index.js';
import Tools from './tools/index.js';
import Effects from './effects/index.js';
import Ui from './ui/index.js';
import Audio from './audio/index.js';
import Demo from './demo/index.js';

const qs = new URLSearchParams(location.search);

const CAMERA_PRESETS = {
  aerial:  { pos: [0, 430, 470],   look: [0, 0, 0] },
  orbit:   { pos: [150, 95, 150],  look: [0, 0, 0] },
  skyline: { pos: [200, 42, 300],  look: [0, 32, 0] },
  street:  { pos: [8, 9, 42],      look: [0, 6, -60] }, // y above the ~4m city plain
  close:   { pos: [26, 15, 28],    look: [0, 12, 0] },
};

const el = (id) => document.getElementById(id);
function showErr(msg) { const n = el('err'); n.style.display = 'block'; n.textContent = String(msg); }
function hideBoot() { const b = el('boot'); if (b) b.remove(); }

async function boot() {
  const three = THREE;
  const seed = qs.has('seed') ? (parseInt(qs.get('seed'), 10) || 0x9e3779b9) : 1337;

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  el('app').appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x8fb2d6); // baseline; environment replaces
  scene.fog = new THREE.Fog(0x8fb2d6, 500, 2600); // baseline aerial perspective

  // far kept tight (just past the sky dome at R=3600): a large far/near ratio
  // wrecks depth precision and makes the terrain z-fight with the water plane.
  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.5, 4000);
  const p0 = CAMERA_PRESETS[qs.get('cam') || 'orbit'] || CAMERA_PRESETS.orbit;
  camera.position.set(...p0.pos);
  camera.lookAt(...p0.look);

  const world = createWorld(seed);
  const events = new EventBus();
  const rng = makeRNG(seed);
  const clock = new Clock();
  if (qs.has('time')) clock.setTimeOfDay(parseFloat(qs.get('time')));
  const perf = new Perf();
  const assets = new Assets(THREE);

  const hemi = new THREE.HemisphereLight(0xbcd4ff, 0x3a3327, 0.7); // baseline; environment augments
  scene.add(hemi);

  const ctx = {
    three, scene, camera, renderer, events, rng, clock, assets, perf, world,
    canvas: renderer.domElement, registry: null,
    render: (s, c) => renderer.render(s, c), // effects may override
  };

  const registry = new Registry(ctx);
  ctx.registry = registry;

  // Registration order = init order (dependencies first).
  [Terrain, Environment, Roads, Zoning, Buildings, Props, Traffic, Simulation,
   Tools, Effects, Ui, Audio, Demo]
    .forEach((M) => registry.register(new M()));

  await registry.initAll();
  events.emit('world:ready', { world });

  const loop = new Loop({ renderer, registry, clock, perf, scene, camera, ctx });

  // --- Showcase staging (gauntlet): one module in a clean scene -------------
  function setModule(name) {
    if (!name) { loop.scene = scene; return null; }
    const mod = registry.get(name);
    const s = new THREE.Scene();
    s.background = new THREE.Color(0x9db6d4);
    s.fog = new THREE.Fog(0x9db6d4, 300, 1600);
    // neutral, modest fill — a warm/bright hemisphere washes saturated albedos
    // (e.g. terrain grass) to a flat tan. The live scene gets real IBL instead.
    s.add(new THREE.HemisphereLight(0xcfe0ff, 0x2b2b26, 0.35));
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(800, 800), assets.material('grass', {}));
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    s.add(ground);
    if (mod && typeof mod.showcase === 'function') {
      try { mod.showcase(s, world, ctx); }
      catch (e) { registry._fail(registry.mods.get(name), 'showcase', e); }
    }
    loop.scene = s;
    camera.position.set(...CAMERA_PRESETS.orbit.pos);
    camera.lookAt(...CAMERA_PRESETS.orbit.look);
    return mod;
  }

  loop.start();
  if (qs.get('module')) setModule(qs.get('module'));

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  const setCameraPreset = (name) => {
    const p = CAMERA_PRESETS[name] || CAMERA_PRESETS.orbit;
    camera.position.set(...p.pos);
    camera.lookAt(...p.look);
  };
  const setTimeOfDay = (t) => { clock.setTimeOfDay(t); events.emit('env:set-time', { t }); };
  const setWeather = (w) => { clock.setWeather(w); events.emit('env:set-weather', { w }); };
  const setSpeed = (tps) => { clock.setTps(tps); };

  window.__APP__ = {
    world, registry, renderer, camera, scene, loop,
    setCameraPreset, setTimeOfDay, setWeather, setModule, setSpeed,
    stats() {
      return {
        fps: perf.fps, drawCalls: perf.drawCalls, triangles: perf.triangles,
        budget: perf.budget, over: perf.overBudget(),
        seed, timeOfDay: clock.t, weather: clock.weather,
        moduleStates: registry.states(),
        moduleErrors: window.__MODULE_ERRORS__ || [],
      };
    },
  };

  requestAnimationFrame(() => { hideBoot(); window.__APP_READY__ = true; });
}

boot().catch((e) => { console.error('[boot]', e); showErr(e && e.stack || e); });
