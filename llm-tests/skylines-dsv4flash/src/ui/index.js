// ---------------------------------------------------------------------------
// ui — HTML overlay: top info bar (city name / population / budget), a live
// time-of-day clock with pause + speed controls, a zone/build tool palette,
// and an auto-fading toast feed. Attached to the fixed #ui-root overlay.
//
// Everything is guarded so the module never throws even if sim/core data or the
// bus is absent. DOM text updates are throttled (~5 Hz); toasts fade via CSS.
// ---------------------------------------------------------------------------

import { bus } from '../core/index.js';

export const id = 'ui';

// ---- tuning ---------------------------------------------------------------
const DAY_REAL_SECONDS = 300;   // full in-game day takes this many real seconds at 1x
const GAME_SPEED = 86400 / DAY_REAL_SECONDS; // game-seconds per real-second @ 1x
const UI_REFRESH_MS = 200;      // cadence for cheap DOM text updates
const TOAST_LIFETIME_MS = 4200; // how long a toast stays before fading out
const MAX_TOASTS = 4;

// Zone/tool palette definition: [mode, label, accent, icon(html)]
const TOOLS = [
  { mode: 'residential', label: 'Residential', accent: '#6fe39b',
    icon: '<path d="M5 11h14l-2.4-7A2 2 0 0 0 14.7 3H9.3a2 2 0 0 0-1.9 1L5 11Z"/><path d="M4 21h16v-8H4z" stroke-width="1.6"/>' },
  { mode: 'commercial', label: 'Commercial', accent: '#62a7ff',
    icon: '<rect x="4" y="3" width="16" height="14" rx="2"/><path d="M12 10a3 3 0 1 0-2.6-1.5"/><circle cx="15.5" cy="8" r="1"/>' },
  { mode: 'industrial', label: 'Industrial', accent: '#ffc24d',
    icon: '<path d="M14 4a3 3 0 0 0-5.2 1.9l-.6 1.1-1.4-.2a3 3 0 0 0-1.7 5.8l1 .3L5.9 15H18l-.2-4.4a5 5 0 0 0-3.4-6.4Z"/><rect x="9" y="14" width="6" height="5"/>' },
  { mode: 'demolish', label: 'Demolish', accent: '#ff7a72',
    icon: '<path d="M7 21h10l1-8H6z"/><rect x="9.5" y="2.5" width="5" height="4" rx="1"/><path d="M12 14v3"/>' },
];

// ---------------------------------------------------------------------------
// Helpers — defensive, never throw.
// ---------------------------------------------------------------------------
function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
}

const pad2 = (n) => String(Math.floor(n)).padStart(2, '0');

function fmtNum(v) {
  if (!Number.isFinite(v)) return '0';
  const r = Math.round(v);
  return r.toLocaleString('en-US');
}

// Money formatter: adapt to magnitude so small sim balances stay readable.
function fmtMoney(v) {
  if (!Number.isFinite(v)) return '$0';
  const abs = Math.abs(v);
  let s;
  if (abs >= 1e6) s = (v / 1e6).toFixed(2) + 'M';
  else if (abs >= 1e4) s = Math.round(v).toLocaleString('en-US');
  else if (abs >= 1000) s = (v / 1e3).toFixed(1) + 'k';
  else if (abs >= 10) s = v.toFixed(0);
  else s = v.toFixed(1);
  const sign = v < 0 ? '-' : '';
  return '$' + sign + s.replace('-', '');
}

function toHHMM(sec) {
  sec = Number.isFinite(sec) ? ((sec % 86400) + 86400) % 86400 : 9 * 3600;
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return pad2(h) + ':' + pad2(m);
}

// ---------------------------------------------------------------------------
// Module state (not world-scoped; purely UI)
// ---------------------------------------------------------------------------
let root = null;        // .skylines-hud container
let els = {};           // cached element refs
let acc = 0;            // refresh accumulator
let speed = 1;          // current speed multiplier (1/2/3); paused read from world.meta
let offs = [];          // bus unsubscribe handles

// ---------------------------------------------------------------------------
// Build the HUD DOM
// ---------------------------------------------------------------------------
function build(world) {
  const hud = el('div', 'skylines-hud');

  // ---- top bar ----
  const topbar = el('header', 'topbar');

  // left: city identity + live stats
  const brand = el('div', 'brand');
  const cityName = el('div', 'city-name', 'Skylines');
  const tagline = el('div', 'tagline', 'NEW CITY · SEED ' + (world?.meta?.seed ?? 1337));
  brand.append(cityName, tagline);

  const stats = el('div', 'city-stats');
  const popChip = statChip('population', () => fmtNum(world?.simulation?.pop), { icon: '<path d="M16 21v-2a4 4 0 0 0-8 0v2"/><circle cx="12" cy="7" r="4"/>' });
  const budgetChip = statChip('budget', () => fmtMoney(world?.budget?.balance ?? world?.simulation?.budget), { icon: '<rect x="3" y="9" width="18" height="11" rx="2"/><circle cx="12" cy="14.5" r="1.8"/>' });
  els.popValue = popChip.chip.querySelector('.chip-value');
  els.budgetValue = budgetChip.chip.querySelector('.chip-value');
  stats.append(popChip.chip, budgetChip.chip);

  // right: clock + transport controls
  const clockBlock = el('div', 'clock-block');
  const timeDisplay = el('div', 'time-display');
  els.timeValue = el('span', 'time-value', toHHMM(world?.meta?.timeOfDaySec));
  const ampm = el('span', 'ampm');
  els.ampm = ampm;
  els.setClockAmPm = () => {
    const sec = world?.meta?.timeOfDaySec ?? 9 * 3600;
    ampm.textContent = (sec % 86400) < 12 * 3600 ? 'AM' : 'PM';
  };
  timeDisplay.append(els.timeValue, ampm);
  els.setClockAmPm();

  const transport = el('div', 'transport');
  // play/pause toggle
  els.pauseBtn = el('button', 'ctl-btn pause-btn', PAUSE_ICON);
  els.pauseBtn.title = 'Play / Pause';
  els.pauseBtn.addEventListener('click', () => {
    if (world?.meta) world.meta.paused = !world.meta.paused;
    syncTransport(world);
  });
  // speed pills
  els.speedBtns = {};
  const speeds = [1, 2, 3];
  speeds.forEach((s) => {
    const b = el('button', 'ctl-btn speed-pill', s + '×');
    b.addEventListener('click', () => {
      speed = s;
      if (world?.meta) world.meta.paused = false;
      syncTransport(world);
    });
    els.speedBtns[s] = b;
    transport.append(b);
  });
  transport.append(els.pauseBtn);

  clockBlock.append(timeDisplay, transport);
  topbar.append(brand, stats, clockBlock);
  hud.append(topbar);

  // ---- left tool palette ----
  const toolbar = el('aside', 'toolbar');
  toolbar.setAttribute('aria-label', 'Build tools');
  els.toolBtns = {};
  TOOLS.forEach((t) => {
    const btn = el('button', 'tool-btn');
    btn.innerHTML = svgIcon(t.icon);
    btn.title = t.label;
    btn.dataset.mode = t.mode;
    try { btn.style.setProperty('--acc', t.accent); } catch (_e) {}
    btn.addEventListener('click', () => {
      if (els.activeTool === t.mode) { setActiveTool(null); return; }
      setActiveTool(t.mode);
      try { bus.emit('tool:mode', { mode: t.mode }); } catch (_e) { /* ignore */ }
    });
    els.toolBtns[t.mode] = btn;
    toolbar.append(btn);
  });
  hud.append(toolbar);

  // ---- toast stack (bottom, centered) ----
  const toasts = el('div', 'toast-stack');
  hud.append(toasts);
  els.toastStack = toasts;

  root = hud;
  return hud;
}

function statChip(label, getter, { icon } = {}) {
  const chip = el('div', 'stat-chip');
  if (icon) chip.append(el('svg', 'chip-icon', svgIcon(icon)));
  const body = el('span', 'chip-body');
  body.append(el('span', 'chip-label', label));
  const valueEl = el('span', 'chip-value', '0');
  chip.getter = getter;
  body.append(valueEl);
  chip.append(body);
  return { chip, valueEl };
}

function syncTransport(world) {
  const paused = !!(world?.meta && world.meta.paused);
  els.pauseBtn.classList.toggle('active', !paused);
  if (els.pauseBtn) els.pauseBtn.innerHTML = paused ? PLAY_ICON : PAUSE_ICON;
  Object.keys(els.speedBtns || {}).forEach((k) => {
    els.speedBtns[k].classList.toggle('active', speed === Number(k));
  });
}

function setActiveTool(mode) {
  els.activeTool = mode;
  Object.keys(els.toolBtns || {}).forEach((m) => {
    els.toolBtns[m].classList.toggle('active', m === mode);
  });
}

// ---- toasts ----------------------------------------------------------------
function pushToast({ text, level }) {
  const stack = els.toastStack;
  if (!stack) return;
  // keep only the freshest few
  while (stack.children.length >= MAX_TOASTS) stack.removeChild(stack.firstChild);

  const t = el('div', 'toast toast-' + (level || 'info'));
  t.textContent = text != null ? String(text) : '';
  stack.appendChild(t);
  requestAnimationFrame(() => t.classList.add('show'));

  // fade out after the lifetime
  setTimeout(() => {
    t.classList.remove('show');
    t.classList.add('out');
    setTimeout(() => { try { if (t.parentNode) t.parentNode.removeChild(t); } catch (_e) {} }, 700);
  }, TOAST_LIFETIME_MS);
}

// ---- public API -------------------------------------------------------------
export function init(world) {
  try {
    const host = document.getElementById('ui-root') || document.body;
    injectStyle(host.ownerDocument);
    const hud = build(world);
    host.appendChild(hud);

    // subscribe to the core event bus (guarded — a no-op bus is fine)
    const b = bus && typeof bus.on === 'function' ? bus : { on() {} };
    offs.push(b.on('ui:toast', pushToast));
    // welcome toast (short-lived)
    setTimeout(() => {
      try { pushToast({ text: 'Welcome, Mayor. Your city awaits.', level: 'info' }); } catch (_e) {}
    }, 700);
    syncTransport(world);
  } catch (e) {
    console.error('[ui] init failed', e);
  }
}

export function update(dt, world) {
  // advance the clock ourselves so it stays live even if no sim module drives TOD
  try {
    const meta = world && world.meta;
    if (meta && typeof meta.timeOfDaySec === 'number' && !meta.paused) {
      const d = Number.isFinite(dt) ? dt : 1 / 20;
      meta.timeOfDaySec = (meta.timeOfDaySec + d * GAME_SPEED * speed) % 86400;
    }
  } catch (_e) { /* ignore */ }

  // throttled DOM refresh
  acc += Number.isFinite(dt) ? dt : 0.016;
  if (acc < UI_REFRESH_MS / 1000) return;
  acc = 0;
  try { refresh(world); } catch (_e) { /* ignore */ }
}

function refresh(world) {
  const s = world && world.simulation ? world.simulation : null;
  // population
  if (els.popValue) els.popValue.textContent = fmtNum(s && s.pop);
  // budget
  const bal = world?.budget?.balance ?? (s && s.budget);
  if (els.budgetValue) {
    els.budgetValue.textContent = fmtMoney(bal);
    els.budgetValue.classList.toggle('neg', Number.isFinite(bal) && bal < 0);
  }
  // clock
  if (els.timeValue) els.timeValue.textContent = toHHMM(world?.meta?.timeOfDaySec);
  if (typeof els.setClockAmPm === 'function') els.setClockAmPm();
}

export function showcase(container) {
  try {
    if (!container || typeof container.appendChild !== 'function') return;
    injectStyle(container.ownerDocument);
    const hud = build({ meta: { timeOfDaySec: 15 * 3600, seed: 1337 }, simulation: { pop: 12480 }, budget: { balance: 482300 } });
    container.appendChild(hud);
    pushToast({ text: 'Showcase build — residential/commercial/industrial/demolish', level: 'info' });
  } catch (_e) { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Icons + styles
// ---------------------------------------------------------------------------
const PAUSE_ICON = '<svg viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/></svg>';
const PLAY_ICON = '<svg viewBox="0 0 24 24"><path d="M8 5.5v13l11-6.5z"/></svg>';

function svgIcon(inner) {
  return `<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round">${inner}</svg>`;
}

const CSS = `
.skylines-hud{position:absolute;inset:0;color:#e8f0f6;font-size:14px;letter-spacing:.01em;
  user-select:none;-webkit-font-smoothing:antialiased;}
.skylines-hud *{box-sizing:border-box;}

/* ---- top bar ---- */
.topbar{position:absolute;top:16px;left:16px;right:16px;display:flex;align-items:flex-start;
  gap:20px;pointer-events:none;}
.brand{display:flex;flex-direction:column;gap:2px;}
.city-name{font-size:22px;font-weight:800;letter-spacing:.02em;color:#f4fbff;
  text-shadow:0 2px 14px rgba(0,0,0,.5);}
.tagline{font-size:11px;font-weight:600;letter-spacing:.22em;text-transform:uppercase;
  color:#7fa3b8;}
.city-stats{display:flex;gap:12px;margin-top:4px;flex-wrap:wrap;}

.stat-chip{display:flex;align-items:center;gap:9px;padding:7px 14px 7px 10px;
  background:linear-gradient(160deg,rgba(16,24,32,.78),rgba(10,16,22,.72));
  border:1px solid rgba(120,190,220,.16);border-radius:12px;
  backdrop-filter:blur(14px) saturate(130%);-webkit-backdrop-filter:blur(14px) saturate(130%);
  box-shadow:0 6px 24px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.06);}
.chip-icon{width:16px;height:16px;color:#63d7e8;}
.chip-body{display:flex;flex-direction:column;gap:0;line-height:1.05;}
.chip-label{font-size:10px;font-weight:600;letter-spacing:.18em;text-transform:uppercase;
  color:#8fb0c4;}
.chip-value{font-size:17px;font-weight:700;color:#eaf6ff;font-variant-numeric:tabular-nums;}
.chip-value.neg{color:#ff7a72;}

/* ---- clock + transport ---- */
.clock-block{margin-left:auto;display:flex;flex-direction:column;align-items:flex-end;gap:8px;
  pointer-events:none;}
.time-display{font-size:34px;font-weight:800;color:#f4fbff;line-height:1;
  text-shadow:0 2px 18px rgba(0,0,0,.55);display:flex;align-items:baseline;gap:6px;}
.ampm{font-size:13px;font-weight:700;letter-spacing:.12em;color:#7fa3b8;}
.transport{display:flex;align-items:center;gap:6px;pointer-events:auto;
  background:rgba(10,16,22,.7);border:1px solid rgba(120,190,220,.16);padding:4px;
  border-radius:11px;backdrop-filter:blur(14px) saturate(130%);-webkit-backdrop-filter:blur(14px) saturate(130%);
  box-shadow:0 6px 22px rgba(0,0,0,.35);}

.ctl-btn{appearance:none;background:transparent;border:1px solid transparent;color:#b9d4e4;
  font-size:13px;font-weight:700;padding:5px 11px;border-radius:8px;cursor:pointer;
  transition:all .15s ease;}
.ctl-btn svg{width:16px;height:16px;display:block;}
.ctl-btn:hover{background:rgba(99,215,232,.12);color:#eaf6ff;}
.ctl-btn:active{transform:translateY(1px);}
.pause-btn.active{color:#eaf6ff;background:rgba(99,215,232,.2);border-color:rgba(99,215,232,.4);}
.speed-pill{padding:5px 12px;}
.speed-pill.active{color:#07242b;background:#63d7e8;box-shadow:0 2px 10px rgba(99,215,232,.45);}

/* ---- left tool palette ---- */
.toolbar{position:absolute;left:16px;top:50%;transform:translateY(-50%);
  display:flex;flex-direction:column;gap:9px;
  padding:8px;background:rgba(10,16,22,.7);border:1px solid rgba(120,190,220,.16);
  border-radius:14px;backdrop-filter:blur(14px) saturate(130%);-webkit-backdrop-filter:blur(14px) saturate(130%);
  box-shadow:0 10px 34px rgba(0,0,0,.4);}
.tool-btn{appearance:none;width:46px;height:46px;display:flex;align-items:center;justify-content:center;
  background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.06);color:#cfe3ef;
  border-radius:11px;cursor:pointer;transition:all .15s ease;}
.tool-btn .ico{width:24px;height:24px;}
.tool-btn:hover{border-color:rgba(120,190,220,.35);color:#eaf6ff;background:rgba(255,255,255,.05);
  transform:translateY(-1px);}
.tool-btn:active{transform:translateY(0) scale(.96);}
.tool-btn.active{border-color:var(--acc,#63d7e8);color:#fff;
  background:color-mix(in srgb,var(--acc,#63d7e8) 26%, transparent);
  box-shadow:0 0 0 1px var(--acc,#63d7e8),0 4px 16px color-mix(in srgb,var(--acc,#63d7e8) 40%, transparent);}

/* ---- toasts ---- */
.toast-stack{position:absolute;left:50%;transform:translateX(-50%);bottom:34px;
  display:flex;flex-direction:column;align-items:center;gap:9px;pointer-events:none;}
.toast{max-width:420px;padding:11px 18px;border-radius:12px;font-size:14px;font-weight:600;
  color:#eaf6ff;text-align:center;opacity:0;transform:translateY(8px);
  transition:opacity .3s ease,transform .3s ease;
  border:1px solid rgba(255,255,255,.14);
  background:linear-gradient(160deg,rgba(16,26,34,.85),rgba(10,16,22,.82));
  backdrop-filter:blur(12px) saturate(120%);-webkit-backdrop-filter:blur(12px) saturate(120%);
  box-shadow:0 8px 28px rgba(0,0,0,.4);}
.toast.show{opacity:1;transform:translateY(0);}
.toast.out{opacity:0;transform:translateY(-6px);}
.toast-info{border-color:rgba(99,215,232,.4);box-shadow:0 8px 28px rgba(0,0,0,.4),0 0 22px rgba(99,215,232,.18);}
.toast-warn{border-color:rgba(255,194,77,.45);color:#ffe9b8;}
.toast-error{border-color:rgba(255,122,114,.5);color:#ffd4d1;}

@media (max-width:760px){
  .time-display{font-size:26px;}
  .brand{display:none;}
}
`;

function injectStyle(doc) {
  try {
    if (!doc || !doc.head) return;
    let s = doc.getElementById('skylines-ui-css');
    if (s) return;
    s = doc.createElement('style');
    s.id = 'skylines-ui-css';
    s.textContent = CSS;
    doc.head.appendChild(s);
  } catch (_e) { /* ignore */ }
}
