// core — shared world data model, event bus, clock, RNG, module loader.
// This is the ONLY folder the integrator may edit freely.
import * as THREE from 'three';

// ---------------------------------------------------------------------------
// Seeded RNG (mulberry32). All modules MUST draw randomness from here.
// ---------------------------------------------------------------------------
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Event bus — global, typed by name. Modules emit namespaced events.
// ---------------------------------------------------------------------------
export const bus = {
  _m: new Map(),
  on(event, fn) {
    if (!this._m.has(event)) this._m.set(event, new Set());
    this._m.get(event).add(fn);
    return () => this.off(event, fn);
  },
  off(event, fn) { this._m.get(event)?.delete(fn); },
  emit(event, data) {
    const set = this._m.get(event);
    if (!set) return;
    for (const fn of [...set]) {
      try { fn(data); } catch (e) { console.error(`[bus] handler error on "${event}"`, e); }
    }
  },
};

// ---------------------------------------------------------------------------
// World — the single serializable source of truth.
// ---------------------------------------------------------------------------
export const world = {
  meta: { name: 'New City', seed: 1337, tick: 0, timeOfDaySec: 9 * 3600, paused: false },
  clock: null,
  rng: mulberry32(1337),
  terrain: { size: 2048, heightAt: null, normalAt: null },
  zones: [],
  buildings: [],
  roads: [],
  props: [],
  agents: [],
  budget: {},
  stats: { fps: 0, drawCalls: 0, frameTimeMs: 0 },
  modules: {},       // id -> module api
  dead: new Set(),   // ids of failed modules (isolation)
};

let _lastFrame = performance.now();
let _fpsFrames = 0;
let _fpsTimer = _lastFrame;

export const clock = {
  simDtSec: 1 / 20,
  _simAcc: 0,
  tick(now) {
    const dt = Math.min((now - _lastFrame) / 1000, 0.1);
    _lastFrame = now;
    // fps estimate
    _fpsFrames++;
    if (now - _fpsTimer >= 1000) {
      world.stats.fps = Math.round(_fpsFrames * 1000 / (now - _fpsTimer));
      _fpsFrames = 0; _fpsTimer = now;
    }
    // fixed-step simulation
    this._simAcc += dt;
    let ticks = 0;
    while (this._simAcc >= this.simDtSec && ticks < 5) {
      world.meta.tick++;
      bus.emit('sim-tick', { tick: world.meta.tick, dt: this.simDtSec });
      this._simAcc -= this.simDtSec;
      ticks++;
    }
    bus.emit('frame', { dtSec: dt });
  },
};

// ---------------------------------------------------------------------------
// Module lifecycle (failure isolation)
// ---------------------------------------------------------------------------
export function registerModule(mod) {
  if (!mod || !mod.id) return null;
  world.modules[mod.id] = mod;
  bus.emit('module:loaded', { id: mod.id });
  return mod;
}

// Explicit import map so vite can statically resolve each module bundle.
const MODULES = {
  terrain: () => import('../terrain/index.js'),
  environment: () => import('../environment/index.js'),
  roads: () => import('../roads/index.js'),
  simulation: () => import('../simulation/index.js'),
  effects: () => import('../effects/index.js'),
  zoning: () => import('../zoning/index.js'),
  buildings: () => import('../buildings/index.js'),
  props: () => import('../props/index.js'),
  traffic: () => import('../traffic/index.js'),
  tools: () => import('../tools/index.js'),
  audio: () => import('../audio/index.js'),
  ui: () => import('../ui/index.js'),
  demo: () => import('../demo/index.js'),
};

export async function loadModule(id) {
  try {
    const loader = MODULES[id];
    if (!loader) throw new Error(`unknown module "${id}"`);
    const m = await loader();
    registerModule(m);
    if (typeof m.init === 'function') m.init(world, world.scene, world.renderer, world.camera);
    return m;
  } catch (e) {
    console.error(`[core] module "${id}" failed to load`, e);
    bus.emit('module:error', { id, error: e?.message || String(e) });
    world.dead.add(id);
    return null;
  }
}

export async function startModules(ids) {
  const results = {};
  for (const id of ids) results[id] = await loadModule(id);
  return results;
}

// update loop: call every frame. Safe per-module; a throw kills only that module this frame.
export function updateModules() {
  for (const [id, mod] of Object.entries(world.modules)) {
    if (world.dead.has(id)) continue;
    try { if (typeof mod.update === 'function') mod.update(clock.simDtSec || 0.016, world); }
    catch (e) {
      console.error(`[core] module "${id}" update error`, e);
      bus.emit('module:error', { id, error: e?.message || String(e) });
      world.dead.add(id);
    }
  }
}

// ---------------------------------------------------------------------------
// Convenience accessors used by screenshot tool + modules
// ---------------------------------------------------------------------------
export function setTimeOfDay(sec) {
  world.meta.timeOfDaySec = sec;
  bus.emit('timeofday:changed', { sec });
}
export function setWeather(id) {
  world.meta.weather = id;
  bus.emit('weather:changed', { id });
}

// Create a fresh world bound to renderer/scene/camera (called from main.js).
export function createWorld({ seed = 1337, renderer, scene, camera } = {}) {
  world.meta.seed = seed;
  world.rng = mulberry32(seed);
  world.scene = scene;
  world.renderer = renderer;
  world.camera = camera;
  return world;
}

// Re-export three for convenience across modules that need geometry helpers.
export { THREE };
