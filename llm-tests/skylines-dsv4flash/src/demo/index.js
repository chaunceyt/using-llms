// src/demo/index.js — Wave 3 showcase. Composes every module into a cinematic,
// self-driving tour of the city: curated vantage points, day/night lighting
// that follows each shot, and a live whole-game dashboard (population, demand,
// budget, fps, draw calls). Loaded last, only via ?demo=1.
//
// The tour yields entirely to world.meta.paused — when paused it neither moves
// the camera nor advances the clock — so the screenshot tool can frame exact
// presets without fighting this module.
import { bus } from '../core/index.js';

export const id = 'demo';

// Curated shots. Each transitions the camera AND the time of day to the preset
// value, then holds for a beat so the lighting can be appreciated.
const TOUR = [
  { name: 'Overview',        pos: [900, 700, 1150], target: [0, 10, 0],     todSec: 15 * 3600 + 20 * 60, hold: 5.0 },
  { name: 'Downtown Street', pos: [70, 9, 40],       target: [260, 6, 0],    todSec: 10 * 3600,           hold: 5.0 },
  { name: 'Skyline Sunset',  pos: [520, 220, 640],   target: [-200, 8, -120], todSec: 18 * 3600 + 12 * 60, hold: 6.0 },
  { name: 'Night City',      pos: [430, 190, 310],   target: [0, 40, 0],     todSec: 21 * 3600 + 30 * 60, hold: 7.0 },
];
const TRANSITION = 3.5; // seconds to dolly between shots

// Module-level state (the ES module namespace is frozen — never assign `this._`).
let _world = null;
let st = {
  running: true,
  seg: 0,
  t: 0,                    // seconds into current segment (transition + hold)
  curPos: TOUR[0].pos.slice(),
  curTarget: TOUR[0].target.slice(),
  ui: null,
  els: {},
};

function easeInOut(t) {
  t = Math.min(1, Math.max(0, t));
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}
function hm(sec) {
  const h = Math.floor((sec % 86400) / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}
function lerp3(out, a, b, k) {
  out[0] = a[0] + (b[0] - a[0]) * k;
  out[1] = a[1] + (b[1] - a[1]) * k;
  out[2] = a[2] + (b[2] - a[2]) * k;
}
function setTime(world, sec) {
  try { bus.emit('timeofday:changed', { sec }); }
  catch (e) { /* never let the tour break on a dead listener */ }
  if (world.meta) world.meta.timeOfDaySec = sec;
}

export function init(world) {
  _world = world;
  st.running = true;
  st.seg = 0;
  st.t = 0;
  st.curPos = TOUR[0].pos.slice();
  st.curTarget = TOUR[0].target.slice();
  bus.emit('demo:ready', { shots: TOUR.length });
}

export function update(dtSec, world) {
  if (!world || !world.camera) return;
  // Respect the pause flag — screenshots frame exact presets while paused.
  if (world.meta.paused) { refreshDashboard(world); return; }
  if (!st.running) { refreshDashboard(world); return; }

  const seg = TOUR[st.seg % TOUR.length];
  const prev = st.t;
  st.t += dtSec;

  if (st.t >= TRANSITION + seg.hold) {
    st.seg++;
    st.t = 0;
    return;
  }
  // Just crossed into the hold — snap lighting to this shot's time of day.
  if (prev < TRANSITION && st.t >= TRANSITION) setTime(world, seg.todSec);

  const k = easeInOut(st.t / TRANSITION);
  const next = TOUR[(st.seg + 1) % TOUR.length];
  lerp3(st.curPos, seg.pos, next.pos, k);
  lerp3(st.curTarget, seg.target, next.target, k);

  world.camera.position.set(st.curPos[0], st.curPos[1], st.curPos[2]);
  world.camera.lookAt(st.curTarget[0], st.curTarget[1], st.curTarget[2]);

  // Gentle forward push while held, so the frame never feels static.
  if (st.t >= TRANSITION && st.t < TRANSITION + seg.hold) {
    world.camera.translateZ(-dtSec * 4);
  }
  refreshDashboard(world);
}

function jumpTo(i) {
  st.seg = i % TOUR.length;
  st.t = TRANSITION + TOUR[st.seg].hold; // land at the end of the dolly
  if (_world) setTime(_world, TOUR[st.seg].todSec);
}

export function showcase(container) {
  const root = document.createElement('div');
  root.style.cssText =
    'position:absolute;top:12px;right:12px;width:250px;font:13px/1.5 system-ui,Segoe UI,sans-serif;' +
    'color:#dfe7ef;background:rgba(8,14,20,.72);backdrop-filter:blur(6px);border:1px solid rgba(255,255,255,.12);' +
    'border-radius:10px;padding:10px 12px;box-shadow:0 6px 24px rgba(0,0,0,.4);pointer-events:auto;';

  const title = document.createElement('div');
  title.style.cssText = 'font-weight:700;letter-spacing:.04em;margin-bottom:2px;';
  title.textContent = (_world?.meta?.name || 'Skylines').replace(/^New /, '') + ' · Demo Tour';
  root.appendChild(title);

  const clockEl = document.createElement('div');
  clockEl.style.cssText = 'color:#ffd27a;font-weight:600;margin-bottom:6px;';
  clockEl.textContent = hm(_world?.meta?.timeOfDaySec ?? 9 * 3600);
  root.appendChild(clockEl);

  const grid = (label) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;justify-content:space-between;gap:10px;';
    const l = document.createElement('span');
    l.textContent = label;
    l.style.color = '#8ea0b5';
    const v = document.createElement('span');
    v.style.color = '#fff';
    row.append(l, v);
    root.appendChild(row);
    return v;
  };
  st.els.pop = grid('Population');
  st.els.jobs = grid('Jobs');
  st.els.demand = grid('Demand');
  st.els.budget = grid('Budget');
  st.els.draw = grid('Draw calls');
  st.els.fps = grid('FPS');

  const bar = document.createElement('div');
  bar.style.cssText = 'display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;';
  const mkBtn = (label, fn) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.style.cssText =
      'flex:1;min-width:70px;padding:4px 6px;font:12px system-ui;color:#d8e2ec;cursor:pointer;' +
      'background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);border-radius:6px;';
    b.onclick = fn;
    return b;
  };
  bar.appendChild(mkBtn('Tour', () => { st.running = !st.running; }));
  TOUR.forEach((shot, i) => {
    const btn = mkBtn(shot.name.split(' ')[0], () => jumpTo(i));
    btn.title = shot.name;
    bar.appendChild(btn);
  });
  root.appendChild(bar);

  window.addEventListener('keydown', (e) => {
    if (e.target && /input|textarea/i.test(e.target.tagName)) return;
    if (e.key === ' ') { e.preventDefault(); st.running = !st.running; }
    else {
      const n = parseInt(e.key, 10);
      if (n >= 1 && n <= TOUR.length) jumpTo(n - 1);
    }
  });

  container.style.cssText += 'pointer-events:auto;';
  container.appendChild(root);
  st.ui = root;
}

function refreshDashboard(world) {
  const els = st.els;
  if (!els.pop) return;
  const sim = world.simulation || {};
  els.pop.textContent = fmt(sim.pop);
  els.jobs.textContent = fmt(sim.jobs);
  els.demand.textContent = sim.demand
    ? `${fmtVal(sim.demand.residential)}/${fmtVal(sim.demand.commercial)}/${fmtVal(sim.demand.industrial)}`
    : '–';
  els.budget.textContent = sim.budget != null ? '$' + Math.round(sim.budget).toLocaleString() : '–';
  els.draw.textContent = String(world.stats?.drawCalls ?? 0);
  els.fps.textContent = world.stats?.fps != null ? world.stats.fps + (world.meta.paused ? ' ⏸' : '') : '–';
  if (st.ui) {
    const clockEl = st.ui.querySelector('div:nth-child(2)');
    if (clockEl && world.meta) clockEl.textContent = hm(world.meta.timeOfDaySec);
  }
}

function fmt(n) { return n == null ? '–' : Number(n).toLocaleString(); }
function fmtVal(n) { return n == null || isNaN(n) ? '–' : String(Math.round(Number(n))); }
