// ============================================================================
// zoning module — Skylines
//
// Builds the zone grid that fills the city's road blocks (the land between
// roads), assigns Residential/Commercial/Industrial districts at density 1..3,
// and grows/densifies/converts cells over sim ticks in response to the economy
// signal `world.simulation.demand.{res,com,ind}`.
//
// Deterministic: every random draw comes from `world.rng` (mulberry32 seeded by
// core). No Math.random. Same seed -> same city, run to run.
//
// Outputs:
//   * world.zones : [{x,z,type,density,growth}] — consumed by simulation +
//     buildings. Emits `zone:changed {cell}` on the core bus whenever a cell
//     changes type or density.
//   * A cheap translucent zoning overlay (3 InstancedMesh quads — green res /
//     blue com / amber ind) so the plan is visible at a glance and the
//     buildings module can read `world.zones` to place on top.
//
// Guarded defensively: if roads are absent it falls back to a plain grid, and
// if simulation is absent growth stays flat. Never throws on boot.
// Draw calls: 3 (one InstancedMesh per district colour).
// ============================================================================
import * as THREE from 'three';
import { bus } from '../core/index.js';

export const id = 'zoning';

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------
const CELL        = 40;          // metres per zone cell (building footprint block)
const HALF_EXTENT = 460;         // zoned region is [-HALF,HALF]^2 on XZ
const ROAD_MARGIN = 12;          // extra clearance beyond road half-width for a buildable cell

// growth / density thresholds
const DEMAND_HIGH = 6;           // demand at/above this actively grows a district
const GROW_SPEED   = 1.7;
const SHRINK_SPEED = 1.3;
const CAP          = 92;         // growth ceiling
// density bands by accumulated growth
function densityOfGrowth(g) {
  if (g > 62) return 3;
  if (g > 30) return 2;
  return 1;
}
// representative starting growth for a given initial density
function seedGrowthFor(density) { return density === 3 ? 74 : density === 2 ? 46 : 15; }

// district colour palette (translucent plan overlay)
const PALETTE = {
  residential:  0x39a05c,   // green
  commercial:   0x3b83d6,   // blue
  industrial:   0xd9a23a,   // amber/yellow
};
// simulation demand key per district type
const DEMAND_KEY = { residential: 'res', commercial: 'com', industrial: 'ind' };
// inverse mapping used for conversions
const TYPE_BY_DEMAND = { res: 'residential', com: 'commercial', ind: 'industrial' };

// ---------------------------------------------------------------------------
// Helpers (defensive; never throw if a piece of world is missing)
// ---------------------------------------------------------------------------
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

function heightAt(world, x, z) {
  try {
    const f = (world && world.terrain && world.terrain.heightAt) || world?.heightAt;
    if (typeof f === 'function') { const h = f(x, z); return Number.isFinite(h) ? h : 0; }
  } catch (_e) { /* terrain absent -> plane */ }
  return 0;
}

// nearest distance from point to a road segment in the XZ plane
function segDist(px, pz, sx, sz, ex, ez) {
  const vx = ex - sx, vz = ez - sz;
  const l2 = vx * vx + vz * vz || 1;
  let t = ((px - sx) * vx + (pz - sz) * vz) / l2;
  t = clamp(t, 0, 1);
  const cx = sx + t * vx, cz = sz + t * vz;
  return Math.sqrt((px - cx) * (px - cx) + (pz - cz) * (pz - cz));
}

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------
const state = {
  built: false,
  cells: [],          // live zone cells {x,z,type,density,growth}
  offTick: null,      // sim-tick subscription handle
  group: null,        // THREE.Group holding the overlay meshes
  unitGeo: null,
  mats: {},           // type -> material
};

// ---------------------------------------------------------------------------
// Grid + carving. Reads road layout from world.roads if present; a cell whose
// centre is within (road half-width + margin) of any segment is treated as
// "road" and excluded, leaving the blocks between roads zoned.
// ---------------------------------------------------------------------------
function buildGrid(world) {
  const cells = [];
  const segs = Array.isArray(world.roads) ? world.roads : [];

  let steps = 0;
  for (let i = -Math.round(HALF_EXTENT / CELL); i <= Math.round(HALF_EXTENT / CELL); i++) {
    for (let j = -Math.round(HALF_EXTENT / CELL); j <= Math.round(HALF_EXTENT / CELL); j++) {
      const cx = i * CELL, cz = j * CELL;
      steps++;
      // carve roads if we have a network
      if (segs.length) {
        let onRoad = false;
        for (const s of segs) {
          const w2 = (Number.isFinite(s.width) ? s.width : 14) / 2 + ROAD_MARGIN;
          const d = segDist(cx, cz, s.from[0], s.from[1], s.to[0], s.to[1]);
          if (d <= w2) { onRoad = true; break; }
        }
        if (onRoad) continue;
      }
      cells.push({ x: cx, z: cz, type: 'vacant', density: 1, growth: 0 });
    }
  }
  return { cells, steps };
}

// Assign a deterministic initial district + density to every buildable cell.
// Districts follow a city plan: industry pushed to the periphery, commerce
// hugging road frontage / the core, residences filling the middle ground.
function assignDistricts(world, cells) {
  const rng = world.rng;
  const indRadius = HALF_EXTENT * 0.62;   // industry beyond this radius is likely
  const comFrontage = CELL * 1.6;         // distance to nearest road that reads as "frontage"

  for (const c of cells) {
    const r = Math.sqrt(c.x * c.x + c.z * c.z);

    // proximity to a road centreline (already known clear of asphalt)
    let nearRoad = 0;
    if (Array.isArray(world.roads)) {
      let best = Infinity;
      for (const s of world.roads) {
        const d = segDist(c.x, c.z, s.from[0], s.from[1], s.to[0], s.to[1]);
        if (d < best) best = d;
      }
      nearRoad = best < comFrontage ? 1 : 0;
    }

    // weighted district choice
    const wRes = 5.0;
    const wCom = r > HALF_EXTENT * 0.85 ? 0.6 : (nearRoad ? 3.2 : 1.0);
    const wInd = r >= indRadius ? 4.5 : (nearRoad && !wRes ? 1.2 : 0.7);

    const tot = wRes + wCom + wInd;
    let roll = rng() * tot;
    let type = 'residential';
    if ((roll -= wRes) < 0) type = 'residential';
    else if ((roll -= wCom) < 0) type = 'commercial';
    else type = 'industrial';

    // density: denser toward the core
    let density = 1;
    if (type !== 'industrial') {
      if (r < HALF_EXTENT * 0.32 && rng() < 0.45) density = 3;
      else if ((r < HALF_EXTENT * 0.6 && rng() < 0.55) || nearRoad) density = 2;
    } else if (r > HALF_EXTENT * 0.8 && rng() < 0.35) {
      density = 2;
    }

    c.type = type;
    c.density = clamp(density, 1, 3);
    c.growth = seedGrowthFor(c.density);
  }
}

// ---------------------------------------------------------------------------
// Overlay — three InstancedMesh quads, one per district colour. Merged + cheap.
// ---------------------------------------------------------------------------
function makeUnitGeo() {
  const g = new THREE.PlaneGeometry(CELL * 0.92, CELL * 0.92);
  g.rotateX(-Math.PI / 2);
  return g;
}

function overlayMaterial(color) {
  return new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.16,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
    side: THREE.DoubleSide,
  });
}

function rebuildOverlay(world) {
  const scene = world.scene;
  // drop the old group WITHOUT disposing shared geo/materials (they live on in
  // state.unitGeo / state.mats and are reused below).
  if (state.group) { scene.remove(state.group); state.group = null; }
  if (!scene) return;

  if (!state.unitGeo) state.unitGeo = makeUnitGeo();
  // reuse materials (stable, cheap)
  for (const t of ['residential', 'commercial', 'industrial']) {
    if (!state.mats[t]) state.mats[t] = overlayMaterial(PALETTE[t]);
  }

  const group = new THREE.Group();
  group.name = 'zoning_overlay';
  const mtx = new THREE.Matrix4();

  for (const t of ['residential', 'commercial', 'industrial']) {
    const members = state.cells.filter(c => c.type === t);
    if (!members.length) continue;
    const inst = new THREE.InstancedMesh(state.unitGeo, state.mats[t], members.length);
    members.forEach((c, k) => {
      mtx.makeTranslation(c.x, heightAt(world, c.x, c.z) + 0.05, c.z);
      inst.setMatrixAt(k, mtx);
    });
    inst.instanceMatrix.needsUpdate = true;
    inst.frustumCulled = false;
    group.add(inst);
  }
  state.group = group;
  scene.add(group);
}

// ---------------------------------------------------------------------------
// Growth step — one deterministic tick of the economy -> city loop.
// Runs on 'sim-tick' (fixed-step), never every frame.
// ---------------------------------------------------------------------------
function growStep(world) {
  const sim = world.simulation;
  if (!state.cells.length || !sim || typeof sim.demand !== 'object') return;

  const d = sim.demand;
  const res = Number.isFinite(d.res) ? d.res : 0;
  const com = Number.isFinite(d.com) ? d.com : 0;
  const ind = Number.isFinite(d.ind) ? d.ind : 0;
  const rng = world.rng;

  let needsRebuild = false;

  for (const c of state.cells) {
    const key = DEMAND_KEY[c.type] || 'res';
    const dem = key === 'com' ? com : key === 'ind' ? ind : res;

    // --- growth / decline from demand ---
    if (dem >= DEMAND_HIGH) {
      c.growth += GROW_SPEED * (0.6 + 0.4 * clamp(dem / 40, 0, 1));
    } else if (dem <= 0) {
      c.growth -= SHRINK_SPEED * (0.5 + 0.5 * clamp(-dem / 15, 0, 1));
    } else {
      // mild drift toward a stable level
      c.growth += (dem / DEMAND_HIGH) * 0.3;
    }
    c.growth = clamp(c.growth, 0, CAP);

    let changed = densityOfGrowth(c.growth) !== c.density;

    // --- occasional conversion when our district is starved and another soars ---
    if (dem < -8 && c.growth <= 12) {
      const bestKey = res >= com && res >= ind ? 'res' : (com >= ind ? 'com' : 'ind');
      const bestVal = bestKey === 'com' ? com : bestKey === 'ind' ? ind : res;
      if (bestVal > DEMAND_HIGH * 1.6 && rng() < 0.04) {
        c.type = TYPE_BY_DEMAND[bestKey];
        c.density = 1;
        c.growth = 8;
        changed = true;
        needsRebuild = true;   // overlay membership shifted -> refresh once after loop
      }
    }

    if (changed) {
      bus.emit('zone:changed', { cell: c });
    }
  }

  if (needsRebuild) rebuildOverlay(world);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
export function init(world) {
  if (state.built || !world.scene) return;
  state.built = true;

  const { cells } = buildGrid(world);
  assignDistricts(world, cells);

  // expose the live zone grid to the rest of the world — state.cells and
  // world.zones are THE SAME objects, so growth/density/type mutations reflect
  // immediately for simulation + buildings without an extra copy.
  state.cells = cells;
  world.zones = cells;

  rebuildOverlay(world);

  // deterministic tick-based growth
  if (typeof state.offTick === 'function') state.offTick();
  state.offTick = bus.on('sim-tick', () => growStep(world));
}

export function update(dtSec, world) {
  void dtSec;
  // Growth advances on sim ticks in growStep(); overlay is static between
  // type changes, so nothing needs doing per frame.
  void world;
}

export function showcase(container) {
  if (!container || typeof container.appendChild !== 'function') return;
  const root = document.createElement('div');
  root.style.cssText =
    'position:absolute;inset:auto auto auto 12px;top:12px;font:13px/1.5 monospace;' +
    'color:#dfe8ef;background:rgba(8,14,20,.72);border:1px solid #24333f;' +
    'padding:10px 12px;border-radius:6px;white-space:pre;min-width:230px;';
  const count = (t) => state.cells.filter(c => c.type === t).length;
  const legend = '<span style="color:#39a05c">■ res</span>  ' +
    '<span style="color:#3b83d6">■ com</span>  ' +
    '<span style="color:#d9a23a">■ ind</span>';
  root.innerHTML =
    `zoning\n${legend}\n` +
    `cells ${state.cells.length}   (res ${count('residential')} · com ${count('commercial')} · ind ${count('industrial')})`;
  container.appendChild(root);
}
