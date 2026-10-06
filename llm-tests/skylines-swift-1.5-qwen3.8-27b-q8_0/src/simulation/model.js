// Pure, deterministic city simulation model.
//
// This is the beating heart of Skylines: economy, population, needs/services,
// land value and district development. It is deliberately rendering-free and
// free of any wall-clock dependency so it is 100% reproducible:
//
//   * The ONLY source of randomness is a seeded RNG fork (core/rng.js).
//   * `tick()` advances a FIXED step (TICK_HOURS of simulated time). No dt,
//     no Date.now(), no Math.random(). Same seed + same number of ticks =>
//     identical city, every time.
//   * Every update is an exponential relaxation toward a target plus hard
//     clamps, so the numbers stay bounded, plausible and NaN-free.
//
// The Simulation module (index.js) wraps this, wires it to the clock (fixed
// step accumulation), events and a minimal Three.js visualisation.

import { makeRNG } from '../core/rng.js';

export const GRID = 8;              // districts per map side (8x8 = 64 districts)
export const TICK_HOURS = 0.25;     // 15 simulated minutes per tick
export const TICKS_PER_DAY = Math.round(24 / TICK_HOURS); // 96

const DH = TICK_HOURS / 24;         // fraction of a day elapsed per tick
const MAP_HALF = 512;               // metres (map is 1024 x 1024)
const CELL = 1024 / GRID;           // 128 m per district

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerp = (a, b, t) => a + (b - a) * t;

// Job multipliers by land use.
const ZONE_JOB = { none: 0.7, residential: 0.5, commercial: 1.7, industrial: 1.2, mixed: 1.0 };

export class SimModel {
  constructor(seed = 1337, opts = {}) {
    this.seed = seed >>> 0;
    this.rng = makeRNG(this.seed).fork('sim'); // deterministic stream
    this.tickNo = 0;
    this.day = 1;

    this.taxRate = clamp(opts.taxRate ?? 0.08, 0, 0.5);
    this.funds = opts.funds ?? 1_000_000;

    const n = GRID * GRID;
    this.cx = new Float64Array(n);
    this.cz = new Float64Array(n);
    this.population = new Float64Array(n);
    this.housing = new Float64Array(n);
    this.housingBonus = new Float64Array(n); // persistent additions (zoning/buildings)
    this.jobs = new Float64Array(n);
    this.jobsBonus = new Float64Array(n);
    this.landValue = new Float64Array(n);
    this.development = new Float64Array(n);
    this.happiness = new Float64Array(n);
    this.powerCov = new Float64Array(n);
    this.waterCov = new Float64Array(n);
    this.serviceCov = new Float64Array(n);
    this.roadProx = new Float64Array(n);
    this.zone = new Array(n).fill('none');

    // city-wide aggregates
    this.totalPopulation = 0;
    this.totalJobs = 0;
    this.satisfaction = 50;         // 0..100
    this.services = 0;              // 0..100 average service coverage
    this.avgLandValue = 0;
    this.demand = { residential: 0, commercial: 0, industrial: 0 }; // 0..100
    this._demandN = { residential: 0, commercial: 0, industrial: 0 }; // 0..1
    this.revenuePerDay = 0;
    this.expensePerDay = 0;
    this.netPerDay = 0;

    this._seedCity();
    this._recomputeAggregates();
  }

  // ---- Seeding a coherent starter city (robust to empty input) -------------
  _seedCity() {
    const n = GRID * GRID;
    for (let iz = 0; iz < GRID; iz++) {
      for (let ix = 0; ix < GRID; ix++) {
        const i = iz * GRID + ix;
        const cx = (ix - GRID / 2 + 0.5) * CELL;
        const cz = (iz - GRID / 2 + 0.5) * CELL;
        this.cx[i] = cx;
        this.cz[i] = cz;
        // 0 at centre, ->1 toward the map edge.
        const d = Math.min(1, Math.hypot(cx, cz) / (MAP_HALF * 0.9));
        const core = 1 - d;
        const jitter = 0.9 + 0.2 * this.rng.next(); // 0.9..1.1, deterministic
        this.development[i] = clamp(0.05 + 0.6 * core * core * jitter, 0, 0.9);
        this.zone[i] = core > 0.72 ? 'mixed' : core > 0.45 ? 'commercial' : 'residential';
        this.housing[i] = 30 + 520 * this.development[i];
        this.jobs[i] = 15 + 260 * this.development[i] * (ZONE_JOB[this.zone[i]] || 1);
        this.population[i] = this.housing[i] * (0.3 + 0.28 * core);
        this.landValue[i] = 120 + 780 * core * core * jitter;
        this.happiness[i] = 55;
        this.roadProx[i] = 0.35 * core + 0.12;
      }
    }
  }

  // ---- One fixed simulation tick -------------------------------------------
  tick() {
    this.tickNo++;
    this._stepDevelopment();
    this._stepServices();
    this._stepHappiness();
    this._stepPopulation();
    this._stepLandValue();
    this._stepEconomy();
    this._recomputeAggregates();
    if (this.tickNo % TICKS_PER_DAY === 0) this.day++;
  }

  // Development creeps toward a target set by land value + demand + funds.
  // Housing capacity and jobs are derived from development (plus bonuses).
  _stepDevelopment() {
    const fundsFactor = clamp(0.35 + 0.65 * (this.funds / 5_000_000), 0.35, 1.1);
    const demand =
      0.5 * this._demandN.residential +
      0.35 * this._demandN.commercial +
      0.15 * this._demandN.industrial;
    const n = GRID * GRID;
    for (let i = 0; i < n; i++) {
      const lvNorm = clamp01(this.landValue[i] / 1500);
      const target = clamp(0.1 + 0.5 * lvNorm + 0.4 * demand * fundsFactor, 0.08, 1);
      const k = 0.02 * (0.7 + 0.6 * target);
      this.development[i] = clamp(this.development[i] + (target - this.development[i]) * k, 0, 1);
      const zf = ZONE_JOB[this.zone[i]] || 1;
      this.housing[i] = 30 + 520 * this.development[i] + this.housingBonus[i];
      this.jobs[i] = 15 + 260 * this.development[i] * zf + this.jobsBonus[i];
    }
  }

  // Service coverage: infrastructure (development + city quality + roads) vs
  // service demand (population + jobs). The demand term is calibrated so a
  // healthy built city settles at ~55-75% average coverage — coverage is a
  // supply/demand ratio, never structurally pinned at 0 or 100.
  _stepServices() {
    const quality = 0.85 + 0.3 * clamp(this.funds / 4_000_000, 0, 1); // 0.85..1.15
    const n = GRID * GRID;
    for (let i = 0; i < n; i++) {
      const infra = (0.2 + 1.4 * this.development[i] + 0.12 * this.roadProx[i]) * quality;
      const load = 0.35 + 0.0048 * (this.population[i] + 0.8 * this.jobs[i]);
      this.powerCov[i] = clamp(infra / load, 0, 1);
      this.waterCov[i] = clamp(infra / (load * 0.92), 0, 1);
      this.serviceCov[i] = clamp(infra / (load * 1.08), 0, 1);
    }
  }

  // Happiness from met needs (services) + job availability + a touch of amenity
  // (land value). Mild pressure terms: a broke city cannot maintain its
  // services, and districts packed past ~85% of housing capacity feel crowded.
  _stepHappiness() {
    const jobCov = clamp(this.totalJobs / Math.max(1, 0.85 * this.totalPopulation), 0, 1);
    const fundPressure = 1 - 0.15 * clamp(Math.max(0, -this.funds) / 1_000_000, 0, 2); // 1.0 .. 0.7
    const n = GRID * GRID;
    for (let i = 0; i < n; i++) {
      const crowd = 1 - 0.2 * clamp((this.population[i] / Math.max(1, this.housing[i]) - 0.85) / 0.15, 0, 1);
      const h =
        100 * (0.28 * this.powerCov[i] + 0.24 * this.waterCov[i] + 0.28 * this.serviceCov[i] + 0.20 * jobCov) *
        fundPressure * crowd +
        6 * (this.landValue[i] / 3000);
      this.happiness[i] = lerp(this.happiness[i], clamp(h, 0, 100), 0.15);
    }
  }

  // Population relaxes toward a fill target driven by vacancy, services,
  // happiness and city-wide job pull. Bounded by housing capacity.
  _stepPopulation() {
    const labor = this.totalJobs / Math.max(1, this.totalPopulation); // >1 = labour shortage
    const jobPull = clamp(0.5 + 0.5 * (labor - 0.8), 0.2, 1.2);
    const n = GRID * GRID;
    for (let i = 0; i < n; i++) {
      const vacancy = Math.max(0, this.housing[i] - this.population[i]) / Math.max(1, this.housing[i]);
      const svc = 0.5 + 0.5 * this.serviceCov[i];
      const happy = 0.5 + 0.5 * (this.happiness[i] / 100);
      const attract = vacancy * svc * happy * (0.6 + 0.4 * jobPull);
      const target = this.housing[i] * (0.45 + 0.5 * attract);
      this.population[i] = clamp(this.population[i] + (target - this.population[i]) * 0.06, 0, this.housing[i]);
    }
  }

  // Land value relaxes toward a target that rewards development, jobs, services
  // and road proximity; demand adds a modest premium.
  _stepLandValue() {
    const demandMul = 1 + 0.4 * (0.5 * this._demandN.residential + 0.5 * this._demandN.commercial);
    const n = GRID * GRID;
    for (let i = 0; i < n; i++) {
      const cov = (this.powerCov[i] + this.waterCov[i] + this.serviceCov[i]) / 3;
      const base = 150 + 1500 * Math.pow(this.development[i], 1.3);
      const target = base * (0.7 + 1.1 * this.roadProx[i]) * (0.6 + 0.6 * cov) * demandMul;
      this.landValue[i] = clamp(this.landValue[i] + (target - this.landValue[i]) * 0.05, 0, 3000);
    }
  }

  // Budget: tax revenue (population x tax rate x per-capita income) vs upkeep +
  // reinvestment of surplus. Per-day rates scaled by the day-fraction per tick.
  _stepEconomy() {
    const pop = this.totalPopulation;
    const avgIncome = 1400 + 1.1 * this.avgLandValue; // $/capita/day, wealthier areas earn more
    const revenue = pop * this.taxRate * avgIncome;
    const upkeep = 80_000 + 95 * pop;
    const surplus = this.funds - 1_000_000;
    const invest = clamp(0.12 * Math.max(0, surplus), 0, 2_500_000);
    const expense = upkeep + invest;
    const net = revenue - expense;
    this.revenuePerDay = revenue;
    this.expensePerDay = expense;
    this.netPerDay = net;
    this.funds = clamp(this.funds + net * DH, -5_000_000, 60_000_000);
  }

  _recomputeAggregates() {
    let pop = 0, jobs = 0, wHap = 0, lv = 0, cov = 0;
    const n = GRID * GRID;
    for (let i = 0; i < n; i++) {
      pop += this.population[i];
      jobs += this.jobs[i];
      wHap += this.happiness[i] * this.population[i];
      lv += this.landValue[i];
      cov += (this.powerCov[i] + this.waterCov[i] + this.serviceCov[i]) / 3;
    }
    this.totalPopulation = pop;
    this.totalJobs = jobs;
    this.satisfaction = pop > 0 ? wHap / pop : this.satisfaction;
    this.avgLandValue = lv / n;
    this.services = (cov / n) * 100;

    const laborShort = clamp((jobs - 0.8 * pop) / (pop + 200), 0, 1);
    const shop = clamp(pop / 8000, 0, 1);
    const ind = clamp(shop * 0.8 + laborShort * 0.2, 0, 1);
    this._demandN.residential = clamp(0.15 + laborShort * 0.85, 0, 1);
    this._demandN.commercial = shop;
    this._demandN.industrial = ind;
    this.demand.residential = Math.round(this._demandN.residential * 100);
    this.demand.commercial = Math.round(shop * 100);
    this.demand.industrial = Math.round(ind * 100);
  }

  // ---- Public queries -------------------------------------------------------
  getPopulation() { return Math.round(this.totalPopulation); }
  getMoney() { return Math.round(this.funds); }
  getHappiness() { return Math.round(this.satisfaction * 10) / 10; }
  getServices() { return Math.round(this.services); }

  // Bilinear sample of land value at world metres (x,z).
  getLandValue(x, z) {
    const gx = (x + MAP_HALF) / CELL - 0.5;
    const gz = (z + MAP_HALF) / CELL - 0.5;
    const x0 = clamp(Math.floor(gx), 0, GRID - 1);
    const z0 = clamp(Math.floor(gz), 0, GRID - 1);
    const x1 = Math.min(GRID - 1, x0 + 1);
    const z1 = Math.min(GRID - 1, z0 + 1);
    const fx = clamp(gx - x0, 0, 1);
    const fz = clamp(gz - z0, 0, 1);
    const a = this.landValue[z0 * GRID + x0] + (this.landValue[z0 * GRID + x1] - this.landValue[z0 * GRID + x0]) * fx;
    const b = this.landValue[z1 * GRID + x0] + (this.landValue[z1 * GRID + x1] - this.landValue[z1 * GRID + x0]) * fx;
    return a + (b - a) * fz;
  }

  getDistrictStats() {
    const out = [];
    for (let i = 0; i < GRID * GRID; i++) {
      out.push({
        ix: i % GRID, iz: (i / GRID) | 0, x: this.cx[i], z: this.cz[i],
        population: Math.round(this.population[i]), housing: Math.round(this.housing[i]),
        jobs: Math.round(this.jobs[i]), landValue: Math.round(this.landValue[i]),
        development: +this.development[i].toFixed(3), happiness: Math.round(this.happiness[i]),
        power: Math.round(this.powerCov[i] * 100), water: Math.round(this.waterCov[i] * 100),
        services: Math.round(this.serviceCov[i] * 100), zone: this.zone[i],
      });
    }
    return out;
  }

  getStats() {
    return {
      population: this.getPopulation(), money: this.getMoney(), happiness: this.getHappiness(),
      ticks: this.tickNo, day: this.day, jobs: Math.round(this.totalJobs),
      services: this.getServices(), landValue: Math.round(this.avgLandValue),
      revenuePerDay: Math.round(this.revenuePerDay), netPerDay: Math.round(this.netPerDay),
      demand: { ...this.demand },
    };
  }

  // ---- Influence setters (for other modules) --------------------------------
  addHousing(n, cell) {
    if (cell == null) {
      let bi = 0, bv = -1;
      for (let i = 0; i < GRID * GRID; i++) {
        const v = this.housing[i] - this.population[i];
        if (v > bv) { bv = v; bi = i; }
      }
      this.housingBonus[bi] += n;
    } else this.housingBonus[clamp(cell, 0, GRID * GRID - 1)] += n;
  }
  addJobs(n, cell) {
    if (cell == null) {
      let bi = 0, bv = -1;
      for (let i = 0; i < GRID * GRID; i++) {
        const v = this.development[i];
        if (v > bv) { bv = v; bi = i; }
      }
      this.jobsBonus[bi] += n;
    } else this.jobsBonus[clamp(cell, 0, GRID * GRID - 1)] += n;
  }
  addPopulation(n) {
    let bi = 0, bv = -1;
    for (let i = 0; i < GRID * GRID; i++) {
      const v = this.housing[i] - this.population[i];
      if (v > bv) { bv = v; bi = i; }
    }
    this.population[bi] = clamp(this.population[bi] + n, 0, this.housing[bi]);
  }
  setZone(cell, use) {
    if (use && ZONE_JOB[use] !== undefined) this.zone[clamp(cell, 0, GRID * GRID - 1)] = use;
  }
  setTaxRate(r) { this.taxRate = clamp(r, 0, 0.5); }
  addFunds(n) { this.funds = clamp(this.funds + n, -5_000_000, 60_000_000); }
  // Road proximity boost (deterministic) near a world position, in cell units.
  // Returns true when any district's proximity actually increased (so callers
  // can tell a live change from an idempotent re-apply).
  bumpRoad(x, z, amount = 0.5, radius = 2) {
    const ccx = (x + MAP_HALF) / CELL - 0.5;
    const ccz = (z + MAP_HALF) / CELL - 0.5;
    const rc = Math.round(ccx), rz = Math.round(ccz);
    let changed = false;
    for (let dz = -radius; dz <= radius; dz++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const ix = rc + dx, iz = rz + dz;
        if (ix < 0 || iz < 0 || ix >= GRID || iz >= GRID) continue;
        const dist = Math.hypot(dx, dz);
        if (dist > radius) continue;
        const fall = Math.min(1, amount * (1 - dist / (radius + 1)));
        const i = iz * GRID + ix;
        if (fall > this.roadProx[i]) { this.roadProx[i] = fall; changed = true; }
      }
    }
    return changed;
  }

  // Map a world position (metres) to a district index.
  cellAt(x, z) {
    const ix = clamp(Math.floor((x + MAP_HALF) / CELL), 0, GRID - 1);
    const iz = clamp(Math.floor((z + MAP_HALF) / CELL), 0, GRID - 1);
    return iz * GRID + ix;
  }
  // Map a 16m tile index to a district index (8 tiles per district).
  cellFromTile(ix, iz) {
    const dx = clamp(Math.floor(ix / 8), 0, GRID - 1);
    const dz = clamp(Math.floor(iz / 8), 0, GRID - 1);
    return dz * GRID + dx;
  }
}
