// Simulation module — the deterministic beating heart of Skylines.
//
// Wraps the pure SimModel (./model.js) with:
//   * fixed-step tick accumulation driven by ctx.clock.tps (frame-rate
//     independent and reproducible: same seed + same ticks => same city),
//   * a deterministic SETTLE burst: a few simulated days of fixed ticks run at
//     boot and after every city change, so the economy reads as ALIVE and
//     settled (real service coverage, satisfaction, running funds balance)
//     even while the clock is paused (tps=0) — e.g. in headless screenshots,
//   * event wiring (roads / zoning / buildings influence the economy),
//   * throttled sim:* events,
//   * a minimal Three.js showcase (a small grid of district "blocks" colored by
//     land value + a floating stats plate) and an optional, OFF-by-default
//     land-value heatmap for the live scene.
//
// Public API: world.sim, tick(dt), getFunds()/getMoney(), getPopulation(),
// getHappiness(), getLandValue(x,z), getDistrictStats(), addHousing(), addJobs(),
// setTaxRate(), addFunds(), setZone(), setHeatmap(on).
// Emits: sim:tick, sim:funds, sim:population, sim:alert. 0 draw calls in the
// live scene (heatmap is opt-in).
import * as THREE from 'three';
import { SimModel, GRID, TICK_HOURS, TICKS_PER_DAY } from './model.js';

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
// Settle burst: ~3.5 simulated days of fixed ticks. Cheap (64-district grid),
// deterministic (no RNG consumed by tick()), and enough for every relaxation
// term (happiness 0.15/tick, population 0.06/tick, development ~0.02-0.03/tick)
// to reach steady state.
const SETTLE_TICKS = 3.5 * TICKS_PER_DAY | 0; // 336
const fmt = (n) => Math.round(n).toLocaleString('en-US');
const fmtMoney = (n) => {
  const v = Math.round(n);
  const sign = v < 0 ? '-' : '';
  const a = Math.abs(v);
  if (a >= 1_000_000) return `${sign}$${(a / 1_000_000).toFixed(2)}M`;
  if (a >= 1_000) return `${sign}$${(a / 1_000).toFixed(0)}k`;
  return `${sign}$${a}`;
};

export default class Simulation {
  name = 'simulation';
  constructor() {
    this._model = null;
    this._subs = [];
    this._acc = 0;
    this._lastPop = 0;
    this._lastFunds = 0;
    this._lastTickEmit = 0;
    this._alertCooldown = 0;
    this._heatMesh = null;
    this._showcaseGroup = null;
    this._dirty = false;               // city changed => re-settle once (not per frame)
    this._settled = false;             // at least one settle burst has run
    this._appliedBuildings = new Set(); // building objects already folded into the model
    this._zonedKeys = new Set();        // zoning tile keys with a capacity bonus applied
  }

  async init(world, ctx) {
    this._ctx = ctx;
    this._world = world;
    this._clock = ctx.clock;
    this._events = ctx.events;
    this._model = new SimModel(world.seed);
    this._lastPop = this._model.getPopulation();
    this._lastFunds = this._model.getMoney();
    this._syncWorldSim();

    const on = (n, cb) => this._subs.push(this._events.on(n, cb));
    on('roads:edge', (e) => this._onRoad(e));
    on('roads:ready', () => this._syncWorldSim());
    on('zoning:changed', (p) => this._onZone(p));
    on('buildings:placed', (p) => this._onBuilding(p));
    on('buildings:ready', () => this._onCityReady());
    on('demo:ready', () => this._onCityReady());

    // The content modules (roads, zoning, buildings) self-place DURING their own
    // init — before we were registered — so their init-time events are already
    // gone. Replay the composed city straight from the shared world state, then
    // settle the economy to steady state so the HUD reads as alive + stable even
    // while the clock is paused (tps=0 in headless screenshots and at boot).
    this._applyWorldContent();
    this._settle();
  }

  // Per-frame: settle pending city changes, then accumulate simulated time and
  // step in fixed ticks.
  update(dt) {
    if (!this._model) return;
    if (this._dirty) this._settle(); // one settle per batch of city changes, even while paused
    const tps = (this._clock && this._clock.tps) || 0;
    if (tps <= 0 || dt <= 0) return; // paused
    this._acc += dt * tps; // simulated hours elapsed
    let steps = 0;
    while (this._acc >= TICK_HOURS && steps < 16) {
      this._acc -= TICK_HOURS;
      this._model.tick();
      steps++;
    }
    // anti spiral-of-death: drop backlog beyond one tick's worth
    if (this._acc >= TICK_HOURS) this._acc = this._acc % TICK_HOURS;
    if (steps > 0) this._postTick();
  }

  // Fixed-step entry (public, mirrors the stub contract). Runs one tick.
  tick() {
    if (!this._model) return;
    this._model.tick();
    this._postTick();
  }

  _postTick() {
    const m = this._model;
    this._syncWorldSim();
    const t = m.tickNo;
    if (t - this._lastTickEmit >= 4) {
      this._lastTickEmit = t;
      this._events.emit('sim:tick', m.getStats());
    }
    const pop = m.getPopulation();
    const funds = m.getMoney();
    if (Math.abs(pop - this._lastPop) > Math.max(50, 0.01 * this._lastPop)) {
      this._events.emit('sim:population', { population: pop, delta: pop - this._lastPop });
      this._lastPop = pop;
    }
    if (Math.abs(funds - this._lastFunds) > 25_000) {
      this._events.emit('sim:funds', { funds, netPerDay: m.netPerDay });
      this._lastFunds = funds;
    }
    this._alerts(m);
  }

  _alerts(m) {
    if (m.tickNo - this._alertCooldown < 24) return; // at most once per ~day
    let alert = null;
    if (m.funds < 0) alert = { type: 'budget', severity: 'high', message: 'City budget in deficit' };
    else if (m.services < 50) alert = { type: 'services', severity: 'medium', message: 'Service coverage below 50%' };
    else if (m.satisfaction < 40) alert = { type: 'unrest', severity: 'medium', message: 'Citizen satisfaction is low' };
    if (alert) {
      this._alertCooldown = m.tickNo;
      this._events.emit('sim:alert', alert);
    }
  }

  _syncWorldSim() {
    const s = this._world.sim;
    const m = this._model;
    if (!s || !m) return;
    s.funds = m.getMoney();
    s.population = m.getPopulation();
    s.jobs = Math.round(m.totalJobs);
    s.services = m.getServices();
    s.satisfaction = m.getHappiness();
    s.demand = { ...m.demand };
    s.day = m.day;
  }

  // ---- Settle: relax the economy to a steady state --------------------------
  // A deterministic burst of fixed ticks (~3.5 simulated days). No Math.random,
  // no wall clock — same seed + same built city => identical settled stats.
  // Cheap: 64-district grid, a few hundred ticks is well under a millisecond
  // per step-pass on modern CPUs.
  _settle() {
    const m = this._model;
    if (!m) return;
    this._dirty = false;
    this._settled = true;
    for (let i = 0; i < SETTLE_TICKS; i++) m.tick();
    this._postTick(); // syncs world.sim and pushes sim:tick / funds / population
  }

  // The composition layer says the city is complete: pick up anything placed
  // after our boot scan (idempotent), then settle — skipped when nothing new
  // arrived so we don't advance the economy a second burst.
  _onCityReady() {
    const added = this._applyWorldContent();
    if (added > 0 || !this._settled) this._settle();
  }

  // Replay the composed city (roads, zoning, buildings) from the shared world
  // state into the model. Idempotent — applied buildings are tracked by
  // reference, zoned tiles by key — so repeat calls never double-count.
  // Returns the number of newly applied zoning/building items.
  _applyWorldContent() {
    const m = this._model, w = this._world;
    if (!m || !w) return 0;
    let added = 0;
    const roads = this._ctx.registry && this._ctx.registry.get('roads');
    if (roads && roads.graph && Array.isArray(roads.graph.edges)) {
      // bumpRoad is a max-assign (idempotent); replay must NOT mark dirty,
      // otherwise a re-scan at demo:ready triggers a second settle burst.
      for (const e of roads.graph.edges) this._bumpRoadEdge(e);
    }
    if (w.zoning && w.zoning.grid) {
      for (const t of w.zoning.grid.values()) added += this._applyZoneTile(t);
    }
    if (Array.isArray(w.buildings)) {
      for (const b of w.buildings) added += this._applyBuilding(b);
    }
    return added;
  }

  // One zoned tile: assign the district use (idempotent) plus a one-time
  // capacity bonus scaled by zoning density. Returns 1 when a bonus was applied.
  _applyZoneTile(t) {
    const m = this._model;
    if (!t || !t.use) return 0;
    const key = t.tx != null && t.tz != null ? t.tx + ',' + t.tz : null;
    const c = t.x != null ? m.cellAt(t.x, t.z) : m.cellFromTile(t.tx, t.tz);
    m.setZone(c, t.use);
    if (key != null && this._zonedKeys.has(key)) return 0; // bonus already applied
    if (key != null) this._zonedKeys.add(key);
    const dn = clamp(((t.density ?? 1) - 1) / 4, 0, 1); // zoning density 1..5 -> 0..1
    const n = Math.round(6 + dn * 22);
    if (t.use === 'commercial' || t.use === 'industrial') m.addJobs(n, c);
    else m.addHousing(n, c);
    return 1;
  }

  // One built structure: capacity in its district, deduped by object reference.
  // Returns 1 when the building was newly applied.
  _applyBuilding(b) {
    const m = this._model;
    if (!b || this._appliedBuildings.has(b)) return 0;
    const x = b.pos ? b.pos.x : b.x;
    const z = b.pos ? b.pos.z : b.z;
    if (x == null || z == null) return 0;
    this._appliedBuildings.add(b);
    const c = m.cellAt(x, z);
    const use = b.use || b.kind;
    if (use === 'residential') m.addHousing(40, c);
    else if (use === 'commercial') m.addJobs(18, c);
    else if (use === 'industrial') m.addJobs(30, c);
    return 1;
  }

  // ---- Event handlers -------------------------------------------------------
  _onRoad(e) {
    if (this._bumpRoadEdge(e)) this._dirty = true;
  }
  // Bump road proximity along one edge. Returns true when the model changed.
  _bumpRoadEdge(e) {
    const a = e && e.a, b = e && e.b;
    if (!a || !b || a.x == null || b.x == null) return false;
    let changed = false;
    for (let t = 0; t <= 1; t += 0.2) {
      const x = a.x + (b.x - a.x) * t, z = a.z + (b.z - a.z) * t;
      if (this._model.bumpRoad(x, z, 0.55, 2)) changed = true;
    }
    return changed;
  }
  _onZone(p) {
    const t = p && p.tile;
    if (!t) return; // boot `{full:true}` payload — covered by the world scan
    this._applyZoneTile({ ...t, use: p.use, density: p.density });
    this._dirty = true;
  }
  _onBuilding(p) {
    if (!p) return;
    if (Array.isArray(p.buildings)) {
      for (const b of p.buildings) this._applyBuilding(b);
    } else {
      this._applyBuilding(p.b || p);
    }
    this._dirty = true;
  }

  // ---- Public query / influence API -----------------------------------------
  getPopulation() { return this._model ? this._model.getPopulation() : 0; }
  getMoney() { return this._model ? this._model.getMoney() : 0; }
  getFunds() { return this.getMoney(); }
  getHappiness() { return this._model ? this._model.getHappiness() : 0; }
  getServices() { return this._model ? this._model.getServices() : 0; }
  getLandValue(x, z) { return this._model ? this._model.getLandValue(x, z) : 0; }
  getDistrictStats() { return this._model ? this._model.getDistrictStats() : []; }
  addHousing(n, cell) { this._model && this._model.addHousing(n, cell); }
  addJobs(n, cell) { this._model && this._model.addJobs(n, cell); }
  addPopulation(n) { this._model && this._model.addPopulation(n); }
  setTaxRate(r) { this._model && this._model.setTaxRate(r); }
  addFunds(n) { this._model && this._model.addFunds(n); }
  setZone(cell, use) { this._model && this._model.setZone(cell, use); }

  // ---- Optional, OFF-by-default live-scene land-value heatmap ---------------
  setHeatmap(on) {
    if (!this._model) return;
    if (on && !this._heatMesh) {
      const three = this._ctx.three;
      this._heatCanvas = document.createElement('canvas');
      this._heatCanvas.width = this._heatCanvas.height = GRID * 8;
      this._heatTex = new three.CanvasTexture(this._heatCanvas);
      this._heatTex.magFilter = three.LinearFilter;
      this._heatTex.minFilter = three.LinearFilter;
      const mat = new three.MeshBasicMaterial({ map: this._heatTex, transparent: true, opacity: 0.45, depthWrite: false });
      const size = 1024;
      this._heatMesh = new three.Mesh(new three.PlaneGeometry(size, size), mat);
      this._heatMesh.rotation.x = -Math.PI / 2;
      this._heatMesh.position.y = 0.6;
      this._heatMesh.renderOrder = 2;
      this._ctx.scene.add(this._heatMesh);
      this._drawHeatmap();
    } else if (!on && this._heatMesh) {
      this._ctx.scene.remove(this._heatMesh);
      this._heatMesh.geometry.dispose();
      this._heatMesh.material.map && this._heatMesh.material.map.dispose();
      this._heatMesh.material.dispose();
      this._heatMesh = null;
      this._heatTex = null;
    }
  }
  _drawHeatmap() {
    const g = this._heatCanvas.getContext('2d');
    const S = this._heatCanvas.width;
    const img = g.createImageData(S, S);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const wx = (x / S - 0.5) * 1024;
        const wz = (y / S - 0.5) * 1024;
        const t = clamp(this._model.getLandValue(wx, wz) / 1600, 0, 1);
        const i = (y * S + x) * 4;
        img.data[i] = Math.round(255 * Math.min(1, t * 2));
        img.data[i + 1] = Math.round(255 * Math.min(1, Math.max(0, 1 - Math.abs(t - 0.5) * 2)));
        img.data[i + 2] = Math.round(255 * Math.min(1, (1 - t) * 2));
        img.data[i + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
    this._heatTex.needsUpdate = true;
  }

  // ---- Showcase: a minimal, readable slice of the simulation ----------------
  showcase(scene, world, ctx) {
    const three = ctx.three;
    // A fresh starter city, advanced a burst of fixed ticks (~6 game days) so the
    // numbers are coherent and spatially varied.
    const m = new SimModel((world.seed ^ 0x51ab3c) >>> 0);
    for (let i = 0; i < 288; i++) m.tick();

    const group = new three.Group();
    this._showcaseModel = m;

    // 1) District blocks: one merged geometry, per-vertex color (land value),
    //    height from development. Sized to fit the orbit camera frame.
    const geo = this._buildCityGeometry(m);
    const mat = new three.MeshStandardMaterial({ vertexColors: true, roughness: 0.82, metalness: 0.0, flatShading: true });
    const city = new three.Mesh(geo, mat);
    city.castShadow = true;
    city.receiveShadow = true;
    group.add(city);

    // 2) Floating stats plate (canvas texture) — the only way to show numbers
    //    in a WebGL-only scene.
    const plate = this._buildTitle(m, three);
    group.add(plate);

    scene.add(group);
    this._showcaseGroup = group;
  }

  _buildCityGeometry(m) {
    const cell = 20; // showcase metres per block
    const pos = [], nrm = [], col = [], idx = [];
    for (let iz = 0; iz < GRID; iz++) {
      for (let ix = 0; ix < GRID; ix++) {
        const i = iz * GRID + ix;
        const x = (ix - GRID / 2 + 0.5) * cell;
        const z = (iz - GRID / 2 + 0.5) * cell;
        const h = 3 + m.development[i] * 44;
        const w = cell * 0.82;
        this._addBox(pos, nrm, col, idx, x, z, w, h);
        // per-vertex color: land value drives the blue->red ramp, happiness the
        // top-face brightness. 5 faces x 4 verts = 20 colors per block.
        const t = clamp(m.landValue[i] / 1600, 0, 1);
        const happy = m.happiness[i] / 100;
        const c = new THREE.Color().setHSL(0.62 - 0.62 * t, 0.7, 0.3 + 0.3 * t);
        const top = c.clone().multiplyScalar(0.7 + 0.5 * happy);
        const side = c.clone().multiplyScalar(0.5);
        for (let f = 0; f < 5; f++) {
          const cc = f === 0 ? top : side;
          for (let k = 0; k < 4; k++) col.push(cc.r, cc.g, cc.b);
        }
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idx);
    g.computeBoundingSphere();
    return g;
  }

  // Append a 5-face box (top + 4 sides, no bottom) to the growing attribute arrays.
  _addBox(pos, nrm, col, idx, cx, cz, w, h) {
    const x0 = cx - w / 2, x1 = cx + w / 2, z0 = cz - w / 2, z1 = cz + w / 2, y0 = 0, y1 = h;
    const faces = [
      [[0, 1, 0], [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
      [[0, 0, 1], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
      [[0, 0, -1], [x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
      [[1, 0, 0], [x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
      [[-1, 0, 0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
    ];
    for (const [n, v0, v1, v2, v3] of faces) {
      const b = idx.length / 3;
      pos.push(v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2]);
      for (let k = 0; k < 4; k++) nrm.push(n[0], n[1], n[2]);
      idx.push(b, b + 1, b + 2, b, b + 2, b + 3);
    }
  }

  _buildTitle(m, three) {
    const cv = document.createElement('canvas');
    cv.width = 1024; cv.height = 256;
    const g = cv.getContext('2d');
    g.fillStyle = 'rgba(9,14,22,0.78)';
    g.fillRect(0, 0, 1024, 256);
    g.fillStyle = '#8fd0ff';
    g.fillRect(0, 0, 1024, 6);
    g.textBaseline = 'top';
    g.fillStyle = '#eaf2ff';
    g.font = 'bold 58px ui-monospace, monospace';
    g.fillText('SKYLINES — CITY SIMULATION', 36, 26);
    g.font = '40px ui-monospace, monospace';
    g.fillStyle = '#cdd9ee';
    g.fillText(`Population  ${fmt(m.getPopulation())}      Funds  ${fmtMoney(m.getMoney())}      Happiness  ${m.getHappiness()}%`, 36, 112);
    g.fillText(`Day  ${m.day}      Jobs  ${fmt(m.totalJobs)}      Services  ${m.getServices()}%      Land $${Math.round(m.avgLandValue)}`, 36, 172);
    const tex = new three.CanvasTexture(cv);
    tex.colorSpace = three.SRGBColorSpace;
    tex.anisotropy = 4;
    const mat = new three.MeshBasicMaterial({ map: tex, transparent: true, side: three.DoubleSide });
    const plane = new three.Mesh(new three.PlaneGeometry(150, 37.5), mat);
    plane.position.set(0, 74, 0);
    plane.lookAt(this._ctx.camera.position);
    return plane;
  }

  stats() {
    const m = this._model;
    return {
      population: m ? m.getPopulation() : 0,
      money: m ? m.getMoney() : 0,
      happiness: m ? m.getHappiness() : 0,
      ticks: m ? m.tickNo : 0,
      districts: GRID * GRID,
      drawCalls: 0,
      notes: m ? `day ${m.day} · funds ${fmtMoney(m.getMoney())} · net ${fmtMoney(m.netPerDay)}/day` : 'stub',
    };
  }

  dispose() {
    this._subs.forEach((off) => off());
    this._subs = [];
    this.setHeatmap(false);
    if (this._showcaseGroup) {
      this._showcaseGroup.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) {
          o.material.map && o.material.map.dispose();
          o.material.dispose();
        }
      });
      this._showcaseGroup = null;
    }
    this._showcaseModel = null;
    this._model = null;
    this._appliedBuildings = new Set();
    this._zonedKeys = new Set();
    this._dirty = false;
    this._settled = false;
  }
}
