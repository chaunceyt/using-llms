// ui — the DOM HUD layered over the canvas. Pure DOM (0 WebGL draw calls); the
// root uses pointer-events: none so the canvas keeps getting drags, with auto
// only on the interactive bits.
//
// Panels:
//   * CITY stats — icon + value + trend (▲/▼) per stat, a funds sparkline and a
//     power/water/sanitation breakdown; hover any row for a detail tooltip.
//   * CITY MAP — a real minimap: height-shaded terrain (water -> grass -> tan ->
//     rock -> snow), the road network, building footprints by class, a district
//     grid, a north marker, and the live camera position + view frustum. The
//     static base is rendered once to an offscreen canvas; the camera overlay is
//     re-composited on camera move and on a 0.5 s cadence.
//   * Time/weather bar (play/pause, speed, clock, weather).
//   * Notification toasts — city charter at boot, daily reports on day change,
//     population milestones, and sim:alert warnings, stacked top-right.
//
// Public API: showPanel(name), setStats(s), setTimeControls(t). Emits ui:action.
// Subscribes to sim:funds / sim:population / sim:tick / sim:alert, world:ready.
// Determinism: every value is derived from world/sim state — no Math.random.
const fmt = (n) => Math.round(n || 0).toLocaleString('en-US');
const fmtMoney = (n) => {
  const v = Math.round(n || 0), sign = v < 0 ? '-' : '', a = Math.abs(v);
  if (a >= 1_000_000) return `${sign}$${(a / 1_000_000).toFixed(2)}M`;
  if (a >= 1_000) return `${sign}$${(a / 1_000).toFixed(0)}k`;
  return `${sign}$${a}`;
};
const clockLabel = (t) => {
  const h = Math.floor(t * 24), m = Math.floor((t * 24 - h) * 60);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
};
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

// ---- inline 13px icons (stroke = currentColor, no font dependency) ----------
const svg = (inner) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${inner}</svg>`;
const svgF = (inner) =>
  `<svg viewBox="0 0 24 24" fill="currentColor" stroke="none" aria-hidden="true">${inner}</svg>`;
const ICONS = {
  funds: svg('<circle cx="12" cy="12" r="8.6"/><path d="M12 7.2v9.6M14.6 9.4c-.5-.9-1.5-1.4-2.6-1.4-1.4 0-2.6.8-2.6 1.9 0 2.5 5.4 1.4 5.4 3.8 0 1.2-1.3 2-2.8 2-1.2 0-2.3-.6-2.8-1.5"/>'),
  pop: svg('<circle cx="12" cy="8.2" r="3.4"/><path d="M5.6 19.4c.9-3.5 3.4-5.2 6.4-5.2s5.5 1.7 6.4 5.2"/>'),
  jobs: svg('<rect x="4" y="8.2" width="16" height="10.8" rx="2"/><path d="M9 8.2V6.6A1.6 1.6 0 0 1 10.6 5h2.8A1.6 1.6 0 0 1 15 6.6v1.6M4 12.8h16"/>'),
  happy: svg('<circle cx="12" cy="12" r="8.6"/><path d="M8.6 14.2c.8 1.3 2 2 3.4 2s2.6-.7 3.4-2M9.2 9.6h.01M14.8 9.6h.01"/>'),
  services: svg('<path d="M15.2 6.4a4.2 4.2 0 0 0-5.8 5L4.6 16.2a1.8 1.8 0 0 0 0 2.5l.7.7a1.8 1.8 0 0 0 2.5 0l4.8-4.8a4.2 4.2 0 0 0 5-5.8l-2.7 2.7-2.2-.5-.5-2.2z"/>'),
  day: svg('<rect x="4" y="6" width="16" height="14.4" rx="2"/><path d="M4 10.4h16M8.4 4v4M15.6 4v4"/>'),
};
const ICO_POWER = svg('<path d="M13 3.2 5.6 13h4.8l-1 7.8L17 11h-4.8z"/>');
const ICO_WATER = svg('<path d="M12 4.2c3 3.8 4.9 6.3 4.9 8.9a4.9 4.9 0 0 1-9.8 0c0-2.6 1.9-5.1 4.9-8.9z"/>');
const ICO_SANIT = svg('<path d="M4.5 10.2c1.8-1.8 3.7-1.8 5.5 0s3.7 1.8 5.5 0 3-1.5 4 0M4.5 14.6c1.8-1.8 3.7-1.8 5.5 0s3.7 1.8 5.5 0 3-1.5 4 0"/>');
const ICO_PAUSE = svgF('<rect x="7" y="5" width="3.4" height="14" rx="1.1"/><rect x="13.6" y="5" width="3.4" height="14" rx="1.1"/>');
const ICO_PLAY = svgF('<path d="M8.2 5.4v13.2a.6.6 0 0 0 .92.5l10.3-6.6a.6.6 0 0 0 0-1L9.12 4.9a.6.6 0 0 0-.92.5z"/>');
const ICO_UP = svgF('<path d="M12 6l7 11H5z"/>');
const ICO_DOWN = svgF('<path d="M12 18L5 7h14z"/>');
const ICO_FLAT = svgF('<rect x="6" y="10.8" width="12" height="2.4" rx="1.2"/>');
const WEATHER_ICONS = {
  clear: svg('<circle cx="12" cy="12" r="4"/><path d="M12 3.5v2M12 18.5v2M20.5 12h-2M5.5 12h-2M18 6l-1.4 1.4M7.4 16.6 6 18M18 18l-1.4-1.4M7.4 7.4 6 6"/>'),
  cloudy: svg('<path d="M6.8 16.5a3.8 3.8 0 0 1-.4-7.58 5.2 5.2 0 0 1 10.05-1.1A4.1 4.1 0 0 1 16.2 16.5z"/>'),
  rain: svg('<path d="M6.8 14.5a3.8 3.8 0 0 1-.4-7.58 5.2 5.2 0 0 1 10.05-1.1A4.1 4.1 0 0 1 16.2 14.5z"/><path d="M8.5 17.5l-1 2.5M12.5 17.5l-1 2.5M16.5 17.5l-1 2.5"/>'),
  fog: svg('<path d="M4.5 10h15M6.5 13.5h13M5.5 17h11"/>'),
};

// Minimap terrain ramp: [height(m), r, g, b]. waterLevel is -3 in the shared
// world; the heightfield spans the 512 m map with a seafloor at ~-33 m.
const MM_RAMP = [
  [-34, 10, 30, 52], [-18, 14, 42, 70], [-8, 20, 58, 88], [-3, 38, 88, 114],
  [-2.1, 122, 152, 138], [-0.6, 182, 166, 116], [2, 122, 146, 84], [12, 78, 116, 56],
  [26, 126, 130, 74], [45, 152, 138, 100], [62, 128, 124, 118], [85, 152, 148, 142],
  [105, 206, 208, 210], [130, 234, 236, 238],
];
function rampAt(h, out) {
  const R = MM_RAMP;
  if (h <= R[0][0]) { out[0] = R[0][1]; out[1] = R[0][2]; out[2] = R[0][3]; return; }
  for (let i = 1; i < R.length; i++) {
    if (h < R[i][0]) {
      const t = (h - R[i - 1][0]) / (R[i][0] - R[i - 1][0]);
      out[0] = R[i - 1][1] + (R[i][1] - R[i - 1][1]) * t;
      out[1] = R[i - 1][2] + (R[i][2] - R[i - 1][2]) * t;
      out[2] = R[i - 1][3] + (R[i][3] - R[i - 1][3]) * t;
      return;
    }
  }
  const L = R[R.length - 1];
  out[0] = L[1]; out[1] = L[2]; out[2] = L[3];
}

const MILESTONES = [1000, 5000, 10000, 25000, 50000, 100000];

const CSS = `
.sk-ui{position:fixed;inset:0;z-index:20;pointer-events:none;
  font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;color:#eaf2ff;}
.sk-ui .panel{pointer-events:auto;position:absolute;background:rgba(10,16,26,.72);
  border:1px solid rgba(120,160,220,.22);border-radius:10px;backdrop-filter:blur(6px);
  box-shadow:0 6px 24px rgba(0,0,0,.35);padding:10px 12px;}
.sk-ui .title{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#8fb4e6;margin-bottom:6px;}
.sk-ui .stats{top:14px;left:14px;min-width:224px;}
.sk-ui .row{display:flex;justify-content:space-between;align-items:center;gap:14px;padding:2px 4px;border-radius:6px;transition:background .12s;}
.sk-ui .row:hover{background:rgba(120,160,220,.14);}
.sk-ui .row .k{color:#9fb2cc;display:flex;align-items:center;gap:6px;}
.sk-ui .row:hover .k{color:#d5e3f7;}
.sk-ui .icn{display:inline-flex;width:13px;height:13px;flex:none;}
.sk-ui .icn svg{width:13px;height:13px;display:block;}
.sk-ui .icn-funds{color:#9fe3b0;} .sk-ui .icn-pop{color:#8fc1ff;} .sk-ui .icn-jobs{color:#ffd28f;}
.sk-ui .icn-happy{color:#ffe08a;} .sk-ui .icn-services{color:#9fd8ff;} .sk-ui .icn-day{color:#aebcd0;}
.sk-ui .rv{display:flex;align-items:center;gap:6px;}
.sk-ui .row .v{font-variant-numeric:tabular-nums;}
.sk-ui .v.neg{color:#ff9a9a;}
.sk-ui .tr{width:10px;display:flex;justify-content:center;flex:none;}
.sk-ui .tr svg{width:9px;height:9px;display:block;}
.sk-ui .tr-pos{color:#8fe3a8;} .sk-ui .tr-neg{color:#ff9a9a;} .sk-ui .tr-flat{color:#5f6f88;}
.sk-ui .div{height:1px;background:rgba(120,160,220,.18);margin:7px 0;}
.sk-ui .slabel{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:#6f85a6;margin:0 0 3px;}
.sk-ui .spark{display:block;width:100%;height:22px;}
.sk-ui .svc{display:flex;align-items:center;gap:6px;padding:1.5px 0;font-size:11px;}
.sk-ui .svc .si{display:inline-flex;width:12px;flex:none;}
.sk-ui .svc .si svg{width:11px;height:11px;display:block;}
.sk-ui .svc .sn{color:#9fb2cc;width:52px;}
.sk-ui .svc .si.pow{color:#ffd76a;} .sk-ui .svc .si.wat{color:#6ac1ff;} .sk-ui .svc .si.san{color:#8fe3a8;}
.sk-ui .svc .bar{flex:1;height:4px;background:rgba(255,255,255,.12);border-radius:2px;overflow:hidden;}
.sk-ui .svc .bar i{display:block;height:100%;border-radius:2px;width:0;transition:width .4s;}
.sk-ui .svc .sv{width:32px;text-align:right;font-variant-numeric:tabular-nums;}
.sk-ui .timebar{pointer-events:auto;position:absolute;left:50%;bottom:16px;transform:translateX(-50%);
  display:flex;align-items:center;gap:10px;background:rgba(10,16,26,.72);
  border:1px solid rgba(120,160,220,.22);border-radius:12px;backdrop-filter:blur(6px);
  padding:8px 14px;box-shadow:0 6px 24px rgba(0,0,0,.35);}
.sk-ui button{pointer-events:auto;cursor:pointer;font:inherit;color:#eaf2ff;
  display:flex;align-items:center;justify-content:center;gap:5px;
  background:rgba(70,110,170,.28);border:1px solid rgba(120,160,220,.3);border-radius:7px;
  padding:5px 10px;transition:background .12s;}
.sk-ui button svg{width:12px;height:12px;display:block;flex:none;}
.sk-ui .wi{display:inline-flex;align-items:center;}
.sk-ui button:hover{background:rgba(90,140,210,.4);}
.sk-ui button.on{background:rgba(90,150,230,.55);border-color:rgba(150,190,255,.6);}
.sk-ui .clock{font-size:16px;font-variant-numeric:tabular-nums;min-width:56px;text-align:center;}
.sk-ui .minimap{top:14px;right:14px;}
.sk-ui .minimap canvas{display:block;border-radius:6px;}
.sk-ui .toasts{position:absolute;top:216px;right:14px;display:flex;flex-direction:column;gap:8px;align-items:flex-end;}
.sk-ui .toast{pointer-events:none;min-width:200px;max-width:252px;background:rgba(10,16,26,.8);
  border:1px solid rgba(120,160,220,.25);border-left:3px solid #6ac1ff;border-radius:8px;
  padding:7px 12px;backdrop-filter:blur(6px);box-shadow:0 6px 20px rgba(0,0,0,.35);
  animation:sk-toast-in .28s ease-out;}
.sk-ui .toast .tt{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#8fb4e6;}
.sk-ui .toast .tb{font-size:12px;color:#eaf2ff;margin-top:2px;line-height:1.45;}
.sk-ui .toast.success{border-left-color:#7fd99a;}
.sk-ui .toast.warning{border-left-color:#ffd76a;}
.sk-ui .toast.danger{border-left-color:#ff7a7a;}
@keyframes sk-toast-in{from{opacity:0;transform:translateX(18px);}to{opacity:1;transform:none;}}
.sk-ui .tip{position:fixed;z-index:40;pointer-events:none;background:rgba(8,13,22,.94);
  border:1px solid rgba(120,160,220,.3);border-radius:8px;padding:8px 11px;font-size:11px;
  line-height:1.55;color:#b9c8e0;opacity:0;transition:opacity .12s;max-width:240px;
  box-shadow:0 8px 24px rgba(0,0,0,.45);}
.sk-ui .tip.show{opacity:1;}
.sk-ui .tip b{color:#eaf2ff;font-weight:600;font-variant-numeric:tabular-nums;}
`;

export default class Ui {
  name = 'ui';
  constructor() {
    this._root = null; this._style = null; this._subs = []; this._els = {};
    this._toasts = [];
    this._hist = { funds: [], pop: [], jobs: [], happy: [], services: [] };
    this._lastDay = null; this._lastSvc = null; this._mileInit = false;
    this._milestones = new Set();
    this._net = null;
    this._svc = { power: 0, water: 0, sanit: 0 };
    // minimap state
    this._mmBase = null;
    this._mmT = 0;
    this._mmCam = { x: NaN, y: NaN, z: NaN, qx: NaN, qy: NaN, qz: NaN, qw: NaN };
    this._mmDir = null;
  }

  async init(world, ctx) {
    this.ctx = ctx; this.world = world;
    this._mmDir = new ctx.three.Vector3();
    this._build();
    const on = (n, cb) => this._subs.push(ctx.events.on(n, cb));
    on('sim:funds', () => this._refresh());
    on('sim:population', () => this._refresh());
    on('sim:tick', (p) => { if (p && p.netPerDay != null) this._net = p.netPerDay; this._refresh(); });
    on('sim:alert', (a) => this._onAlert(a));
    on('world:ready', () => this._bootToast());
    // world data (terrain heights, roads, buildings) is composed before ui inits
    this._buildMinimapBase();
    this._refresh();
  }

  _build() {
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);
    this._style = style;

    const root = document.createElement('div');
    root.className = 'sk-ui';

    // ---- stats panel ---------------------------------------------------------
    const stats = document.createElement('div');
    stats.className = 'panel stats';
    stats.innerHTML =
      `<div class="title">City</div>` +
      this._row('funds', 'Funds') +
      this._row('pop', 'Population') +
      this._row('jobs', 'Jobs') +
      this._row('happy', 'Happiness') +
      this._row('services', 'Services') +
      this._row('day', 'Day') +
      `<div class="div"></div>` +
      `<div class="slabel">Treasury trend</div>` +
      `<canvas class="spark" width="196" height="22"></canvas>` +
      `<div class="div"></div>` +
      this._svcRow('pow', ICO_POWER, 'Power', 'power') +
      this._svcRow('wat', ICO_WATER, 'Water', 'water') +
      this._svcRow('san', ICO_SANIT, 'Sanit.', 'sanit');
    root.appendChild(stats);

    // ---- minimap panel ---------------------------------------------------------
    const mm = document.createElement('div');
    mm.className = 'panel minimap';
    mm.innerHTML = `<div class="title">City Map</div>`;
    const mmCanvas = document.createElement('canvas');
    // 2x backing store for a crisp 148 css-px map
    mmCanvas.width = mmCanvas.height = 296;
    mmCanvas.style.width = mmCanvas.style.height = '148px';
    mm.appendChild(mmCanvas);
    root.appendChild(mm);

    // ---- time bar ---------------------------------------------------------------
    const bar = document.createElement('div');
    bar.className = 'timebar';
    const play = document.createElement('button');
    play.innerHTML = ICO_PAUSE; play.title = 'Play / pause';
    const speeds = [1, 2, 4].map((s) => {
      const b = document.createElement('button');
      b.textContent = s + '×'; b.dataset.speed = s;
      return b;
    });
    const clock = document.createElement('div');
    clock.className = 'clock'; clock.textContent = '00:00';
    const weather = document.createElement('button');
    weather.innerHTML = `<span class="wi">${WEATHER_ICONS.clear}</span> clear`;
    bar.append(play, ...speeds, clock, weather);
    root.appendChild(bar);

    // ---- toast stack + tooltip ---------------------------------------------------
    const toasts = document.createElement('div');
    toasts.className = 'toasts';
    root.appendChild(toasts);
    const tip = document.createElement('div');
    tip.className = 'tip';
    root.appendChild(tip);

    document.body.appendChild(root);
    this._root = root;
    this._els = {
      funds: stats.querySelector('[data-v="funds"]'),
      pop: stats.querySelector('[data-v="pop"]'),
      jobs: stats.querySelector('[data-v="jobs"]'),
      happy: stats.querySelector('[data-v="happy"]'),
      services: stats.querySelector('[data-v="services"]'),
      day: stats.querySelector('[data-v="day"]'),
      trFunds: stats.querySelector('[data-tr="funds"]'),
      trPop: stats.querySelector('[data-tr="pop"]'),
      trJobs: stats.querySelector('[data-tr="jobs"]'),
      trHappy: stats.querySelector('[data-tr="happy"]'),
      trServices: stats.querySelector('[data-tr="services"]'),
      rows: {
        funds: stats.querySelector('[data-row="funds"]'),
        pop: stats.querySelector('[data-row="pop"]'),
        jobs: stats.querySelector('[data-row="jobs"]'),
        happy: stats.querySelector('[data-row="happy"]'),
        services: stats.querySelector('[data-row="services"]'),
        day: stats.querySelector('[data-row="day"]'),
      },
      spark: stats.querySelector('.spark'),
      svc: {
        power: { bar: stats.querySelector('[data-svc="power"] .bar i'), v: stats.querySelector('[data-svc="power"] .sv') },
        water: { bar: stats.querySelector('[data-svc="water"] .bar i'), v: stats.querySelector('[data-svc="water"] .sv') },
        sanit: { bar: stats.querySelector('[data-svc="sanit"] .bar i'), v: stats.querySelector('[data-svc="sanit"] .sv') },
      },
      clock, play, weather, toasts, tip, mmCanvas,
    };
    this._sparkCtx = this._els.spark.getContext('2d');
    this._mmCtx = mmCanvas.getContext('2d');

    // hover tooltips on stat rows
    for (const id of Object.keys(this._els.rows)) {
      this._els.rows[id].addEventListener('mouseenter', () => this._showTip(id));
      this._els.rows[id].addEventListener('mouseleave', () => this._hideTip());
    }

    // ---- time controls wiring ---------------------------------------------------
    const clk = this.ctx.clock;
    const baseTps = 24 / 60; // one sim day per 60s at 1×
    this._baseTps = baseTps; this._speed = 1;
    play.addEventListener('click', () => {
      this._paused = !this._paused;
      clk.setTps(this._paused ? 0 : this._speed * baseTps);
      play.innerHTML = this._paused ? ICO_PLAY : ICO_PAUSE;
      play.classList.toggle('on', !this._paused);
      this.ctx.events.emit('ui:action', { action: 'toggle-play', paused: this._paused });
    });
    play.classList.add('on');
    // the button starts in the "playing" state — make the clock match it, so
    // the city (and the HUD's trends/sparkline/daily reports) stays live
    clk.setTps(this._speed * baseTps);
    speeds.forEach((b) => b.addEventListener('click', () => {
      this._speed = Number(b.dataset.speed);
      speeds.forEach((x) => x.classList.toggle('on', x === b));
      if (!this._paused) clk.setTps(this._speed * baseTps);
      this.ctx.events.emit('ui:action', { action: 'set-speed', speed: this._speed });
    }));
    speeds[0].classList.add('on');
    const weathers = ['clear', 'cloudy', 'rain', 'fog'];
    weather.addEventListener('click', () => {
      const i = (weathers.indexOf(clk.weather) + 1) % weathers.length;
      clk.setWeather(weathers[i]);
      weather.innerHTML = `<span class="wi">${WEATHER_ICONS[weathers[i]]}</span> ${weathers[i]}`;
      this.ctx.events.emit('ui:action', { action: 'set-weather', weather: weathers[i] });
    });
  }

  _row(id, label) {
    return `<div class="row" data-row="${id}">` +
      `<span class="k"><span class="icn icn-${id}">${ICONS[id]}</span>${label}</span>` +
      `<span class="rv"><span class="v" data-v="${id}">—</span><span class="tr tr-flat" data-tr="${id}"></span></span>` +
      `</div>`;
  }

  _svcRow(cls, icon, label, id) {
    return `<div class="svc" data-svc="${id}">` +
      `<span class="si ${cls}">${icon}</span><span class="sn">${label}</span>` +
      `<span class="bar"><i></i></span><span class="sv">—</span></div>`;
  }

  // ---- stats refresh ----------------------------------------------------------
  _refresh() {
    const s = this.world.sim;
    if (!s || !this._els.funds) return;
    const e = this._els;
    e.funds.textContent = fmtMoney(s.funds);
    e.funds.classList.toggle('neg', s.funds < 0);
    e.pop.textContent = fmt(s.population);
    e.jobs.textContent = fmt(s.jobs);
    e.happy.textContent = Math.round(s.satisfaction) + '%';
    e.services.textContent = Math.round(s.services) + '%';
    e.day.textContent = s.day;

    // history + trends + sparkline
    const h = this._hist;
    const push = (k, v) => { const a = h[k]; a.push(v); if (a.length > 24) a.shift(); };
    push('funds', s.funds); push('pop', s.population); push('jobs', s.jobs);
    push('happy', s.satisfaction); push('services', s.services);
    this._setTrend(e.trFunds, h.funds, 5000);
    this._setTrend(e.trPop, h.pop, 25);
    this._setTrend(e.trJobs, h.jobs, 10);
    this._setTrend(e.trHappy, h.happy, 0.5);
    this._setTrend(e.trServices, h.services, 0.5);
    this._drawSpark();

    // services breakdown (mean district coverage from the sim model)
    const sim = this.ctx.registry && this.ctx.registry.get('simulation');
    if (sim && typeof sim.getDistrictStats === 'function') {
      const ds = sim.getDistrictStats();
      let p = 0, w = 0, sn = 0;
      for (const d of ds) { p += d.power; w += d.water; sn += d.services; }
      const n = ds.length || 1;
      this._svc = { power: p / n, water: w / n, sanit: sn / n };
      for (const k of ['power', 'water', 'sanit']) {
        e.svc[k].bar.style.width = `${clamp(this._svc[k], 0, 100).toFixed(0)}%`;
        e.svc[k].v.textContent = `${Math.round(this._svc[k])}%`;
      }
    }

    // daily report on day change
    if (this._lastDay != null && s.day > this._lastDay) this._dailyReport(s);
    this._lastDay = s.day;
    if (this._lastSvc == null) this._lastSvc = s.services;

    // population milestones (only crossings after boot)
    if (!this._mileInit) {
      for (const m of MILESTONES) if (s.population >= m) this._milestones.add(m);
      this._mileInit = true;
    }
    for (const m of MILESTONES) {
      if (s.population >= m && !this._milestones.has(m)) {
        this._milestones.add(m);
        this._toast('success', 'City milestone', `Population reached ${fmt(m)} citizens`, 14);
      }
    }
  }

  _setTrend(el, arr, eps) {
    if (!el) return;
    if (arr.length < 3) { el.innerHTML = ''; return; }
    const d = arr[arr.length - 1] - arr[0];
    el.innerHTML = d > eps ? ICO_UP : d < -eps ? ICO_DOWN : ICO_FLAT;
    el.className = 'tr ' + (d > eps ? 'tr-pos' : d < -eps ? 'tr-neg' : 'tr-flat');
  }

  _drawSpark() {
    const g = this._sparkCtx;
    if (!g) return;
    const a = this._hist.funds;
    const W = 196, H = 22;
    g.clearRect(0, 0, W, H);
    if (a.length < 2) return;
    let mn = Infinity, mx = -Infinity;
    for (const v of a) { if (v < mn) mn = v; if (v > mx) mx = v; }
    if (mx - mn < 1) { mn -= 1; mx += 1; }
    const up = a[a.length - 1] >= a[0];
    const col = up ? '#8fe3a8' : '#ff9a9a';
    g.beginPath();
    for (let i = 0; i < a.length; i++) {
      const x = 2 + (i / (a.length - 1)) * (W - 4);
      const y = H - 3 - ((a[i] - mn) / (mx - mn)) * (H - 6);
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    g.strokeStyle = col; g.lineWidth = 1.5; g.stroke();
    g.lineTo(W - 2, H - 1); g.lineTo(2, H - 1); g.closePath();
    g.fillStyle = up ? 'rgba(143,227,168,0.12)' : 'rgba(255,154,154,0.12)';
    g.fill();
    // last-value dot
    const lx = W - 2, ly = H - 3 - ((a[a.length - 1] - mn) / (mx - mn)) * (H - 6);
    g.beginPath(); g.arc(lx, ly, 1.8, 0, 6.2832); g.fillStyle = col; g.fill();
  }

  // ---- notifications ----------------------------------------------------------
  _toast(sev, title, body, ttl = 10) {
    if (!this._els.toasts) return;
    const t = document.createElement('div');
    t.className = `toast ${sev}`;
    const tt = document.createElement('div'); tt.className = 'tt'; tt.textContent = title;
    const tb = document.createElement('div'); tb.className = 'tb'; tb.textContent = body;
    t.append(tt, tb);
    this._els.toasts.appendChild(t);
    this._toasts.push({ el: t, age: 0, ttl });
    while (this._toasts.length > 4) this._toasts.shift().el.remove();
  }

  _toastUpdate(dt) {
    for (let i = this._toasts.length - 1; i >= 0; i--) {
      const t = this._toasts[i];
      t.age += dt;
      const remain = t.ttl - t.age;
      if (remain <= 0) { t.el.remove(); this._toasts.splice(i, 1); continue; }
      if (remain < 1.2) t.el.style.opacity = String(Math.max(0, remain / 1.2));
    }
  }

  _bootToast() {
    const s = this.world.sim;
    if (!s) return;
    this._toast('info', 'City charter',
      `Day ${s.day} · population ${fmt(s.population)} · services ${Math.round(s.services)}%`, 30);
  }

  _dailyReport(s) {
    let extra = '';
    if (this._lastSvc != null) {
      const d = Math.round(s.services) - Math.round(this._lastSvc);
      if (d > 0) extra = ` · services up to ${Math.round(s.services)}%`;
      else if (d < 0) extra = ` · services down to ${Math.round(s.services)}%`;
    }
    this._lastSvc = s.services;
    const netS = this._net == null ? '' :
      ` · net ${this._net >= 0 ? '+' : '-'}${fmtMoney(Math.abs(this._net))}/day`;
    this._toast('info', `Daily report — day ${s.day}`, `Population ${fmt(s.population)}${netS}${extra}`, 12);
  }

  _onAlert(a) {
    if (!a || !a.message) return;
    const sev = a.severity === 'high' ? 'danger' : 'warning';
    const titles = { budget: 'Budget alert', services: 'Services alert', unrest: 'Citizen unrest' };
    this._toast(sev, titles[a.type] || 'Alert', a.message, 14);
  }

  // ---- tooltips -----------------------------------------------------------------
  _showTip(id) {
    const tip = this._els.tip;
    const s = this.world.sim;
    if (!tip || !s) return;
    const c = this._svc;
    const body = {
      funds: `Treasury <b>${fmtMoney(s.funds)}</b>${this._net == null ? '' : `<br>Net <b>${this._net >= 0 ? '+' : '-'}${fmtMoney(Math.abs(this._net))}</b> per day`}`,
      pop: `Citizens <b>${fmt(s.population)}</b><br>Jobs <b>${fmt(s.jobs)}</b> · ${(s.population > 0 ? (s.jobs / s.population).toFixed(2) : '0.00')}/citizen`,
      jobs: `Jobs <b>${fmt(s.jobs)}</b><br>Demand C/I <b>${s.demand ? s.demand.commercial : 0}/${s.demand ? s.demand.industrial : 0}</b>`,
      happy: `Satisfaction <b>${Math.round(s.satisfaction)}%</b><br>Services <b>${Math.round(s.services)}%</b> met`,
      services: `Coverage <b>${Math.round(s.services)}%</b><br>Power <b>${Math.round(c.power)}%</b> · Water <b>${Math.round(c.water)}%</b> · Sanit. <b>${Math.round(c.sanit)}%</b>`,
      day: `Day <b>${s.day}</b><br>${clockLabel(this.ctx.clock.t)} · ${this.ctx.clock.weather}`,
    }[id];
    tip.innerHTML = body || '';
    const r = this._els.rows[id].getBoundingClientRect();
    tip.style.top = `${Math.max(4, r.top - 8)}px`;
    tip.style.left = `${r.right + 14}px`;
    tip.classList.add('show');
  }

  _hideTip() {
    if (this._els.tip) this._els.tip.classList.remove('show');
  }

  // ---- minimap ------------------------------------------------------------------
  // Static base: terrain (height + hillshade), district grid, roads, buildings,
  // north marker. Rendered once to an offscreen 296² canvas.
  _buildMinimapBase() {
    const S = 296;
    const cv = document.createElement('canvas');
    cv.width = cv.height = S;
    const g = cv.getContext('2d');
    const tn = this.world.terrain;
    const { res, heights, size, waterLevel } = tn;
    const n = res + 1, cell = size / res, half = size / 2;
    const span = 512; // map covers the full 512 m terrain extent
    const sc = S / span;

    const sampleH = (x, z) => {
      let gx = (x + half) / cell, gz = (z + half) / cell;
      if (gx < 0) gx = 0; else if (gx > res - 1e-4) gx = res - 1e-4;
      if (gz < 0) gz = 0; else if (gz > res - 1e-4) gz = res - 1e-4;
      const x0 = gx | 0, z0 = gz | 0, fx = gx - x0, fz = gz - z0;
      const i00 = z0 * n + x0, i10 = i00 + 1, i01 = i00 + n, i11 = i01 + 1;
      const a = heights[i00] + (heights[i10] - heights[i00]) * fx;
      const b = heights[i01] + (heights[i11] - heights[i01]) * fx;
      return a + (b - a) * fz;
    };

    // 1) height grid over the canvas
    const H = new Float32Array(S * S);
    for (let py = 0; py < S; py++) {
      const z = (py / S - 0.5) * span;
      for (let px = 0; px < S; px++) {
        H[py * S + px] = sampleH((px / S - 0.5) * span, z);
      }
    }

    // 2) colour pass: elevation ramp + hillshade (light from the NW)
    const img = g.createImageData(S, S);
    const d = img.data;
    const c = [0, 0, 0];
    const pxStep = span / S; // metres per pixel
    for (let py = 0; py < S; py++) {
      for (let px = 0; px < S; px++) {
        const i = py * S + px;
        const h = H[i];
        rampAt(h, c);
        let shade;
        if (h <= waterLevel + 0.3) {
          shade = 1; // flat water: no relief shading
        } else {
          const xl = H[py * S + Math.max(0, px - 1)], xr = H[py * S + Math.min(S - 1, px + 1)];
          const zu = H[Math.max(0, py - 1) * S + px], zd = H[Math.min(S - 1, py + 1) * S + px];
          const dhdx = (xr - xl) / (2 * pxStep), dhdz = (zd - zu) / (2 * pxStep);
          // surface normal ∝ (-dhdx, 1, -dhdz); light dir NW sky ≈ (-0.45, 0.85, -0.28)
          const dot = dhdx * 0.45 + 0.85 + dhdz * 0.28;
          const inv = 1 / Math.hypot(dhdx, 1, dhdz);
          shade = clamp(0.6 + 0.55 * dot * inv, 0.42, 1.22);
        }
        const o = i * 4;
        d[o] = clamp(c[0] * shade, 0, 255);
        d[o + 1] = clamp(c[1] * shade, 0, 255);
        d[o + 2] = clamp(c[2] * shade, 0, 255);
        d[o + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);

    // 3) faint 8×8 district grid
    g.strokeStyle = 'rgba(255,255,255,0.07)';
    g.lineWidth = 1;
    for (let i = 1; i < 8; i++) {
      const p = (i / 8) * S;
      g.beginPath(); g.moveTo(p, 0); g.lineTo(p, S); g.stroke();
      g.beginPath(); g.moveTo(0, p); g.lineTo(S, p); g.stroke();
    }

    // 4) road network
    const roads = this.ctx.registry && this.ctx.registry.get('roads');
    const wx = (x) => (x / span + 0.5) * S;
    g.lineCap = 'round'; g.lineJoin = 'round';
    if (roads && roads.graph) {
      for (const e of roads.graph.edges) {
        const pts = e.pts;
        if (!pts || pts.length < 2) continue;
        g.beginPath();
        g.moveTo(wx(pts[0].x), wx(pts[0].z));
        for (let i = 1; i < pts.length; i++) g.lineTo(wx(pts[i].x), wx(pts[i].z));
        if (e.kind === 'a') { g.strokeStyle = 'rgba(255,232,170,0.9)'; g.lineWidth = 2.6; }
        else { g.strokeStyle = 'rgba(232,238,248,0.55)'; g.lineWidth = 1.5; }
        g.stroke();
      }
    }

    // 5) building footprints by class
    const bs = this.world.buildings;
    if (Array.isArray(bs)) {
      for (const b of bs) {
        const col = b.kind === 'commercial' ? 'rgba(168,212,255,0.9)'
          : b.kind === 'industrial' ? 'rgba(214,142,96,0.9)'
          : 'rgba(232,198,144,0.85)';
        g.fillStyle = col;
        const w = Math.max(2, (b.w || 7) * sc * 0.8);
        const h = Math.max(2, (b.d || 7) * sc * 0.8);
        g.fillRect(wx(b.x) - w / 2, wx(b.z) - h / 2, w, h);
      }
    }

    // 6) north marker (top-left) + inner border
    g.fillStyle = 'rgba(255,255,255,0.9)';
    g.beginPath(); g.moveTo(14, 5); g.lineTo(9, 14); g.lineTo(19, 14); g.closePath(); g.fill();
    g.font = 'bold 12px ui-monospace, monospace';
    g.textAlign = 'center'; g.textBaseline = 'top';
    g.fillText('N', 14, 17);
    g.textAlign = 'left';
    g.strokeStyle = 'rgba(140,180,240,0.3)';
    g.lineWidth = 1;
    g.strokeRect(0.5, 0.5, S - 1, S - 1);

    this._mmBase = cv;
  }

  // Composite: base + live camera position/frustum.
  _drawMinimap() {
    const g = this._mmCtx;
    if (!g) return;
    const S = 296, span = 512;
    g.clearRect(0, 0, S, S);
    if (this._mmBase) g.drawImage(this._mmBase, 0, 0);
    const cam = this.ctx.camera;
    if (!cam) return;
    const px = (cam.position.x / span + 0.5) * S;
    const pz = (cam.position.z / span + 0.5) * S;
    cam.getWorldDirection(this._mmDir);
    let fx = this._mmDir.x, fz = this._mmDir.z;
    const l = Math.hypot(fx, fz);
    if (l > 0.001) { fx /= l; fz /= l; }
    const sx = -fz, sz = fx; // right vector in XZ
    const A = 0.5, ca = Math.cos(A), sa = Math.sin(A), L = 54;
    g.beginPath();
    g.moveTo(px, pz);
    g.lineTo(px + (fx * ca + sx * sa) * L, pz + (fz * ca + sz * sa) * L);
    g.lineTo(px + (fx * ca - sx * sa) * L, pz + (fz * ca - sz * sa) * L);
    g.closePath();
    g.fillStyle = 'rgba(126,200,255,0.16)';
    g.fill();
    g.strokeStyle = 'rgba(126,200,255,0.6)';
    g.lineWidth = 1.5;
    g.stroke();
    g.beginPath(); g.arc(px, pz, 4.2, 0, 6.2832);
    g.fillStyle = '#eaf6ff'; g.fill();
    g.beginPath(); g.arc(px, pz, 2.1, 0, 6.2832);
    g.fillStyle = '#4da3ff'; g.fill();
  }

  // ---- per-frame ---------------------------------------------------------------
  update(dt, world) {
    const clk = this.ctx.clock;
    if (this._els.clock) this._els.clock.textContent = clockLabel(clk.t);
    this._toastUpdate(dt);

    // minimap: redraw on camera move, on a 0.5 s cadence
    this._mmT -= dt;
    const cam = this.ctx.camera;
    if (cam) {
      const c = this._mmCam, p = cam.position, q = cam.quaternion;
      const moved =
        Math.abs(p.x - c.x) > 0.05 || Math.abs(p.y - c.y) > 0.05 || Math.abs(p.z - c.z) > 0.05 ||
        Math.abs(q.x - c.qx) > 1e-4 || Math.abs(q.y - c.qy) > 1e-4 ||
        Math.abs(q.z - c.qz) > 1e-4 || Math.abs(q.w - c.qw) > 1e-4;
      if (moved) {
        c.x = p.x; c.y = p.y; c.z = p.z;
        c.qx = q.x; c.qy = q.y; c.qz = q.z; c.qw = q.w;
        this._mmT = 0.5;
      }
    }
    if (this._mmT <= 0) { this._mmT = 0.5; this._drawMinimap(); }
  }

  // ---- public API -----------------------------------------------------------
  showPanel(name) {
    this.ctx.events.emit('ui:action', { action: 'show-panel', panel: name });
  }
  setStats(s) {
    if (s) { Object.assign(this.world.sim, s); this._refresh(); }
  }
  setTimeControls(t) {
    if (t && t.t != null) this.ctx.clock.setTimeOfDay(t.t);
    if (t && t.weather) this.ctx.clock.setWeather(t.weather);
  }

  dispose() {
    this._subs.forEach((off) => off());
    this._subs = [];
    for (const t of this._toasts) t.el.remove();
    this._toasts = [];
    if (this._root) this._root.remove();
    if (this._style) this._style.remove();
    this._root = this._style = null; this._els = {};
    this._mmBase = null; this._sparkCtx = null; this._mmCtx = null; this._mmDir = null;
  }

  stats() {
    return { drawCalls: 0, notes: 'DOM HUD: stats (icons/trends/sparkline/services) + real minimap + toasts + tooltips' };
  }
}
