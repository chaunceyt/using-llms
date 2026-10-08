// ---------------------------------------------------------------------------
// simulation — fixed-tick economic / citizen simulator.
//
// Pure logic: models residential/commercial/industrial demand driven by
// population and jobs, and tracks city budget. Produces a `world.simulation`
// signal that zoning/traffic modules can consume later.
//
// Heavy work runs only on sim ticks (bus 'sim-tick'), never every frame.
// All randomness draws from `world.rng` (seeded mulberry32) — no Math.random —
// so the city evolves identically for a fixed seed. Guarded against missing
// zones/buildings: this module must never throw if other modules are absent.
// ---------------------------------------------------------------------------

import { bus, mulberry32 } from '../core/index.js';

export const id = 'simulation';

// subscription handle for the fixed-step clock (kept at module scope because
// the ES module namespace is frozen and cannot accept properties)
let _offTick = null;

// Private seeded PRNG derived from world.meta.seed. Deliberately NOT the shared
// world.rng stream: other modules also draw from world.rng during boot/ticks,
// so sharing it would make our numbers depend on their consumption order. A
// private mulberry32 seeded from the seed keeps this module fully reproducible
// for a fixed seed regardless of what other modules do.
let _rng = null;

// ---- economic tuning constants -------------------------------------------------
const SIM_DT             = 1 / 20;      // seconds per sim tick (matches core clock)
const POP_PER_UNIT       = 4;           // residents per residential building unit
const JOBS_PER_COM       = 3;           // jobs per commercial unit
const JOBS_PER_IND       = 5;           // jobs per industrial unit
const EMPLOYMENT_RATIO   = 0.9;         // ~90% of jobs filled by residents (natural unemployment)
const BASE_RESIDENTS     = 12;          // baseline households even with zero jobs
const MIGRATION_SPEED    = 0.0025;      // fraction of the housing gap closed per tick
const FRICTION           = 0.06;        // frictional unemployment while matching pop->jobs
const COMMERCE_NEED      = 0.6;         // commercial jobs demanded per resident
const INDUSTRY_NEED      = 0.35;        // industrial jobs demanded per resident
const EXPORT_BASE        = 5;           // baseline external demand for industry
const DEMAND_SCALE       = 8;           // converts a workforce surplus into demand units
const DEMAND_MIN         = -15;
const DEMAND_MAX         = 40;

// taxes / upkeep (per tick, kept small so the balance evolves slowly)
const RES_TAX_PER_POP    = 0.012;
const COM_TAX_PER_JOB    = 0.02;
const IND_TAX_PER_JOB    = 0.015;
const UPKEEP_BASE        = 0.5;
const UPKEEP_PER_POP     = 0.002;

// milestone toast throttle
const TOAST_INTERVAL_TICKS = 90;        // min ticks between toasts
const MILESTONE_STEP       = 100;       // population threshold for a toast

// ---------------------------------------------------------------------------
// Helpers — defensive reads so the sim never throws if other modules are absent.
// ---------------------------------------------------------------------------
function safeArray(a) { return Array.isArray(a) ? a : []; }

// Sum the number of "units" contributed by buildings+zones whose type contains `tag`.
function countType(world, tag) {
  let n = 0;
  for (const b of safeArray(world.buildings)) {
    if (b && typeof (b.type || '') === 'string' && b.type.toLowerCase().includes(tag)) {
      n += Number.isFinite(b.units) ? Math.max(1, Math.round(b.units)) : 1;
    }
  }
  for (const z of safeArray(world.zones)) {
    if (z && typeof (z.type || '') === 'string' && z.type.toLowerCase().includes(tag)) {
      const g = Number.isFinite(z.growth) ? z.growth : 0;
      if (g > 0) n += 1;
    }
  }
  return n;
}

function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---------------------------------------------------------------------------
// init — register the fixed-step handler; seeds deterministic initial state.
// Called as init(world, scene, renderer, camera); only world is used here.
// ---------------------------------------------------------------------------
export function init(world) {
  // Ensure our home lives on the shared world object even before the first tick.
  if (!world.simulation) world.simulation = {};

  // private seeded RNG for this module (see note at module scope)
  if (_rng === null) {
    const seed = Number.isFinite(world.meta && world.meta.seed) ? world.meta.seed : 0;
    _rng = mulberry32(seed);
  }

  const sim = world.simulation;
  if (typeof sim.pop !== 'number') {
    // deterministic small founding population from the seed
    const initPop = Math.floor(_rng() * 18) + 6;
    Object.assign(sim, {
      pop: initPop,
      jobs: 0,
      employed: 0,
      demand: { res: 2, com: 1, ind: 1 },
      budget: 0,
      tick: 0,
      lastToastTick: -Infinity,
    });
  }

  // Subscribe to core's fixed-step clock. Runs exactly once per sim tick (up to
  // 5/frame) regardless of fps — the heavy work never runs every frame.
  if (world.meta && typeof world.meta === 'object') {
    if (typeof _offTick === 'function') _offTick();   // no double subscriptions on re-init
    _offTick = bus.on('sim-tick', () => { step(world); });
  }
}

// ---------------------------------------------------------------------------
// step — one deterministic economic tick.
// ---------------------------------------------------------------------------
export function step(world) {
  if (!world || !world.simulation) return;
  // respect pause if core ever wires it up; sim-tick still fires, so bail early
  if (world.meta && world.meta.paused) return;

  const sim = world.simulation;
  if (_rng === null) _rng = mulberry32(Number.isFinite(world.meta && world.meta.seed) ? world.meta.seed : 0);
  const rng = _rng;

  // ---- capacity / jobs from whatever zones+buildings exist today ----
  const resUnits = countType(world, 'res');
  const comUnits = countType(world, 'com');
  const indUnits = countType(world, 'ind');
  const resSlots = resUnits * POP_PER_UNIT;
  const comJobs  = comUnits * JOBS_PER_COM;
  const indJobs  = indUnits * JOBS_PER_IND;
  const jobs     = comJobs + indJobs;

  // ---- migration: residents drift toward the employment-supported level ----
  let targetPop = Math.max(BASE_RESIDENTS, jobs * EMPLOYMENT_RATIO);
  if (resSlots > 0) targetPop = Math.min(targetPop, resSlots); // housing caps growth
  sim.pop += (targetPop - sim.pop) * MIGRATION_SPEED;
  if (sim.pop < 0.5) sim.pop = 0.5;

  // ---- employment matching (with frictional unemployment) ----
  sim.jobs = jobs;
  const available = Math.min(sim.pop, jobs);
  sim.employed = Math.max(0, available * (1 - FRICTION));

  // ---- demand signals consumed later by the zoning module ----
  const resSurplus = jobs * EMPLOYMENT_RATIO + BASE_RESIDENTS - sim.pop;
  const comSurplus = sim.pop * COMMERCE_NEED - comJobs;
  const indSurplus = sim.pop * INDUSTRY_NEED + EXPORT_BASE - indJobs;
  // small deterministic noise so the signal breathes
  const nRes = (rng() - 0.5) * 3;
  const nCom = (rng() - 0.5) * 3;
  const nInd = (rng() - 0.5) * 2;

  sim.demand.res = clamp(Math.round(resSurplus / DEMAND_SCALE + nRes), DEMAND_MIN, DEMAND_MAX);
  sim.demand.com = clamp(Math.round(comSurplus / DEMAND_SCALE + nCom), DEMAND_MIN, DEMAND_MAX);
  sim.demand.ind = clamp(Math.round(indSurplus / DEMAND_SCALE + nInd), DEMAND_MIN, DEMAND_MAX);

  // ---- budget: taxes on activity minus upkeep; accumulates ----
  const income = sim.pop * RES_TAX_PER_POP
    + comJobs * COM_TAX_PER_JOB
    + indJobs * IND_TAX_PER_JOB;
  const upkeep = UPKEEP_BASE + sim.pop * UPKEEP_PER_POP + resUnits * 0.02 + jobs * 0.008;
  const delta = income - upkeep;
  sim.budget += delta;

  // mirror onto the shared budget slot for other modules / UI
  if (world.budget && typeof world.budget === 'object') {
    world.budget.balance = sim.budget;
    world.budget.delta = delta;
  }

  sim.tick = world.meta ? world.meta.tick : sim.tick;

  maybeToast(world, sim);
}

// ---------------------------------------------------------------------------
// maybeToast — lightweight milestone notifications (throttled).
// ---------------------------------------------------------------------------
export function maybeToast(world, sim) {
  const popFloor = Math.floor(sim.pop);
  if (popFloor > 0 && popFloor % MILESTONE_STEP === 0) {
    if (sim.tick - sim.lastToastTick >= TOAST_INTERVAL_TICKS) {
      sim.lastToastTick = sim.tick;
      // no listeners? emit is a safe no-op; keeps us on the core event contract.
      bus.emit('ui:toast', { text: `Population reached ${popFloor}`, level: 'info' });
    }
  }
}

// ---------------------------------------------------------------------------
// update — per-frame hook required by the contract. Heavy work happens in
// step() on sim ticks only, so this stays effectively free.
// ---------------------------------------------------------------------------
export function update(_dtSec, world) {
  // no per-frame simulation work; state advances on 'sim-tick' (fixed-step).
  void world;
}

// ---------------------------------------------------------------------------
// showcase — render current sim numbers into the given container (never global
// page DOM). Falls back silently if container is missing or not an element.
// ---------------------------------------------------------------------------
export function showcase(container) {
  if (!container || typeof container.appendChild !== 'function') return;

  const root = document.createElement('div');
  root.style.cssText =
    'position:absolute;inset:auto auto auto 12px;top:12px;font:13px/1.5 monospace;' +
    'color:#dfe8ef;background:rgba(8,14,20,.72);border:1px solid #24333f;' +
    'padding:10px 12px;border-radius:6px;white-space:pre;min-width:230px;';
  root.textContent = 'simulation: …';

  container.appendChild(root);

  const fmt = (n, d = 0) => {
    if (!Number.isFinite(n)) return '–';
    return n.toFixed(d);
  };
  const refresh = () => {
    const s = world && world.simulation ? world.simulation : null;
    if (!s) { root.textContent = 'simulation: (inactive)'; return; }
    const d = s.demand || {};
    root.textContent =
      `tick        ${world.meta ? world.meta.tick : s.tick}\n` +
      `population  ${fmt(s.pop, 1)}\n` +
      `jobs        ${fmt(s.jobs, 0)} (${fmt(s.employed, 1)} employed)\n` +
      `demand res  ${fmt(d.res, 0)}\n` +
      `demand com  ${fmt(d.com, 0)}\n` +
      `demand ind  ${fmt(d.ind, 0)}\n` +
      `budget      ${fmt(s.budget, 2)}`;
  };
  refresh();
  const timer = setInterval(refresh, 200);
  // best-effort cleanup when the container is detached
  if (typeof MutationObserver !== 'undefined') {
    try {
      const observer = new MutationObserver(() => {
        if (!container.isConnected) { clearInterval(timer); observer.disconnect(); }
      });
      observer.observe(document.body, { childList: true });
    } catch (_e) { /* ignore */ }
  }
}
