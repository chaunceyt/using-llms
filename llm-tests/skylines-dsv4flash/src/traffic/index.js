// ============================================================================
// traffic module — Skylines
// Deterministic car + pedestrian agents flowing along the road network.
//
// Owns ONLY src/traffic/. Everything here is a pure function of world.meta.seed.
//
// Model:
//   * Build a directed lane graph from world.roads segments. Each straight road
//     yields two one-way lanes (right-hand traffic) offset from the centreline,
//     split into pieces at crossings so cars can continue straight or turn onto
//     a crossing road at junctions. Road ends become natural U-turn turnaround
//     nodes, so no agent ever stalls.
//   * Cars follow arc-length along an edge each frame; on reaching a node they
//     pick a forward-aligned outgoing lane (straight preferred, turns allowed)
//     chosen with a deterministic hash of (carId,nodeId) — no world.rng consumed
//     at runtime, so the sim is fully reproducible.
//   * A handful of pedestrians walk on the sidewalk (beyond each lane's outer
//     edge), sharing the same node graph but a larger lateral offset.
//
// Visuals:
//   * ONE InstancedMesh for all car bodies (merged body+cabin boxes, per-instance
//     paint colours). ONE InstancedMesh for headlights(white)/taillights(red)
//     emissive glows that ramp up at night. ONE InstancedMesh for pedestrians.
//     => traffic contributes ~3 draw calls (well within its ~80 share).
//   * Distance culling: each frame agents are sorted near->far and the instance
//     count is trimmed to those within a radius, so far cars stop rendering.
//
// Guards: if roads is absent/empty this module builds a small ring-road loop demo
// so it can still be screenshotted. Never throws (all calls defended).
// ============================================================================
import * as THREE from 'three';
import { bus } from '../core/index.js';

export const id = 'traffic';

// ---------------------------------------------------------------------------
// Deterministic helpers (mulberry32 clone; independent of world.rng draw order)
// ---------------------------------------------------------------------------
function mulberry32(a0) {
  let a = a0 >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
// deterministic hash -> [0,1), independent of frame timing
function randFrom(seed) {
  let n = seed >>> 0;
  n = Math.imul(n ^ (n >>> 16), 2246822519);
  n = Math.imul(n ^ (n >>> 13), 3266489917);
  return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
}

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------
const state = {
  built: false,
  isDemo: false,
  edges: [],                    // directed lane edges
  nodes: new Map(),             // nodeKey -> { x,z, out:[edgeIdx], in:[edgeIdx] }
  cars: [],
  peds: [],
  carMesh: null, lightMesh: null, pedMesh: null,
  edgeCountCached: -1,
};

const CAR_COLOR_PAL = [
  0xd8dbe2, 0x30343a, 0x9aa5ad, 0xb0392b, 0x2f4b7c, 0xe6c229,
  0xffffff, 0x5a5d63, 0x1d7874, 0x8e5573, 0xcd7f32, 0x425b66,
];
const PED_COLOR_PAL = [
  0xc0392b, 0x2980b9, 0x27ae60, 0xf39c12, 0x8e44ad, 0x34495e,
  0xe67e22, 0x16a085, 0x7f8c8d, 0xd35400, 0x1abc9c, 0x795548,
];

const CAR_CULL = 950;    // metres — beyond this cars are not instanced
const PED_CULL = 320;

// temp objects (reused to avoid per-frame allocation)
const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _eul = new THREE.Euler(0, 0, 0);
const _one = new THREE.Vector3(1, 1, 1);
const _m4 = new THREE.Matrix4();

// ---------------------------------------------------------------------------
// Geometry: merge a list of primitive boxes into one non-indexed geometry
// ---------------------------------------------------------------------------
function mergeBoxes(boxes) {
  const pos = [], nrm = [], uv = [];
  for (const b of boxes) {
    const g = new THREE.BoxGeometry(b.w, b.h, b.l).translate(b.x || 0, b.y || 0, b.z || 0);
    const ni = g.toNonIndexed();
    pos.push(...ni.attributes.position.array);
    nrm.push(...ni.attributes.normal.array);
    uv.push(...ni.attributes.uv.array);
    g.dispose();
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  return geo;
}

function carGeometry() {
  // origin at ground centre; nose faces +Z
  return mergeBoxes([
    { w: 1.9, h: 0.55, l: 4.3, y: 0.5 },              // body
    { w: 1.66, h: 0.5, l: 2.15, y: 1.02, z: -0.35 },   // cabin
    { w: 1.8, h: 0.06, l: 4.05, y: 0.79 },             // shoulder/waist
  ]);
}

function lightGeometry() {
  return mergeBoxes([{ w: 0.22, h: 0.12, l: 0.07, y: 0.72 }]);
}

function pedGeometry() {
  return mergeBoxes([
    { w: 0.46, h: 1.25, l: 0.32, y: 0.75 },            // torso
    { w: 0.27, h: 0.27, l: 0.27, y: 1.62 },            // head
  ]);
}

// ---------------------------------------------------------------------------
// Terrain height guard
// ---------------------------------------------------------------------------
function heightAt(world, x, z) {
  try {
    const f = world && world.heightAt;
    if (typeof f === 'function') { const v = f(x, z); return isFinite(v) ? v : 0; }
  } catch (e) { /* fall through */ }
  try {
    const f2 = world && world.terrain && world.terrain.heightAt;
    if (typeof f2 === 'function') { const v = f2(x, z); return isFinite(v) ? v : 0; }
  } catch (e) { /* fall through */ }
  return 0;
}

// ---------------------------------------------------------------------------
// Lane graph construction
// ---------------------------------------------------------------------------
const nodeKey = (x, z) => `${Math.round(x * 10)},${Math.round(z * 10)}`;

function addNode(id, x, z) {
  if (!state.nodes.has(id)) state.nodes.set(id, { x, z, out: [], in: [] });
}
function pushEdge(e) {
  const idx = state.edges.length;
  state.edges.push(e);
  const f = state.nodes.get(e.from), t = state.nodes.get(e.to);
  if (f) f.out.push(idx);
  if (t) t.in.push(idx);
}

// car centre point at fraction t along edge (centreline + off * right(dir))
function edgeCarPos(e, t) {
  const cx = e.cx0 + (e.cx1 - e.cx0) * t;
  const cz = e.cz0 + (e.cz1 - e.cz0) * t;
  return { x: cx - e.dz * e.off, z: cz + e.dx * e.off };
}
// pedestrian point on the sidewalk beyond the lane
function edgePedPos(e, t) {
  const cx = e.cx0 + (e.cx1 - e.cx0) * t;
  const cz = e.cz0 + (e.cz1 - e.cz0) * t;
  const o = Math.sign(e.off) * (e.width / 2 + 1.7);
  return { x: cx - e.dz * o, z: cz + e.dx * o };
}

function buildGraph(segs) {
  state.edges.length = 0;
  state.nodes.clear();

  // crossings between segment pairs (correct 2D intersection test)
  const cross = segs.map(() => []);
  for (let i = 0; i < segs.length; i++) {
    for (let j = i + 1; j < segs.length; j++) {
      const A = segs[i], B = segs[j];
      const DaX = A.bx - A.ax, DaZ = A.bz - A.az;
      const DbX = B.bx - B.ax, DbZ = B.bz - B.az;
      const denom = DaX * DbZ - DaZ * DbX;          // cross(Da,Db)
      if (Math.abs(denom) < 1e-9) continue;
      const dX = B.ax - A.ax, dZ = B.az - A.az;
      const t = (dX * DbZ - dZ * DbX) / denom;      // cross(d,Db)/denom
      if (t < 1e-3 || t > 1 - 1e-3) continue;
      const u = (dX * DaZ - dZ * DaX) / denom;      // cross(d,Da)/denom
      if (u < 1e-3 || u > 1 - 1e-3) continue;
      const cx = A.ax + t * (A.bx - A.ax), cz = A.az + t * (A.bz - A.az);
      cross[i].push({ t, x: cx, z: cz });
      cross[j].push({ t: u, x: cx, z: cz });
    }
  }

  for (let i = 0; i < segs.length; i++) {
    const S = segs[i];
    const len = Math.hypot(S.bx - S.ax, S.bz - S.az) || 1;
    const dx = (S.bx - S.ax) / len, dz = (S.bz - S.az) / len;

    // centreline node list ascending in arc
    const nl = [{ arc: 0, x: S.ax, z: S.az }];
    for (const c of cross[i]) nl.push({ arc: c.t * len, x: c.x, z: c.z });
    nl.push({ arc: len, x: S.bx, z: S.bz });
    nl.sort((a, b) => a.arc - b.arc);
    const ded = [nl[0]];
    for (let k = 1; k < nl.length; k++) if (nl[k].arc - ded[ded.length - 1].arc > 0.5) ded.push(nl[k]);

    const off = S.width / 4;

    // travel A -> B
    for (let k = 0; k < ded.length - 1; k++) {
      const a = ded[k], b = ded[k + 1];
      const fid = nodeKey(a.x, a.z), tid = nodeKey(b.x, b.z);
      addNode(fid, a.x, a.z); addNode(tid, b.x, b.z);
      const eLen = Math.hypot(b.x - a.x, b.z - a.z);
      pushEdge({ from: fid, to: tid, cx0: a.x, cz0: a.z, cx1: b.x, cz1: b.z, dx, dz, off: +off, width: S.width, len: eLen });
    }
    // travel B -> A
    for (let k = ded.length - 1; k > 0; k--) {
      const a = ded[k], b = ded[k - 1];
      const fid = nodeKey(a.x, a.z), tid = nodeKey(b.x, b.z);
      addNode(fid, a.x, a.z); addNode(tid, b.x, b.z);
      const eLen = Math.hypot(b.x - a.x, b.z - a.z);
      pushEdge({ from: fid, to: tid, cx0: a.x, cz0: a.z, cx1: b.x, cz1: b.z, dx: -dx, dz: -dz, off: -off, width: S.width, len: eLen });
    }
  }
}

function spawnCars(rng) {
  const N = Math.min(280, state.edges.length * 2 + 30);
  state.cars.length = 0;
  for (let i = 0; i < N; i++) {
    const e = state.edges[Math.floor(rng() * state.edges.length)];
    if (!e) continue;
    const col = new THREE.Color(CAR_COLOR_PAL[Math.floor(rng() * CAR_COLOR_PAL.length)]);
    // believable city speed ~ 25..45 km/h with a few faster
    const speed = (7 + rng() * 6) * (rng() < 0.12 ? 1.5 : 1.0);
    state.cars.push({ id: i, edge: e, s: rng() * (e.len || 1), speed, color: col });
  }
}

function spawnPeds(rng) {
  const M = Math.min(110, Math.floor(state.edges.length * 0.8));
  state.peds.length = 0;
  for (let i = 0; i < M; i++) {
    const e = state.edges[Math.floor(rng() * state.edges.length)];
    if (!e) continue;
    const col = new THREE.Color(PED_COLOR_PAL[Math.floor(rng() * PED_COLOR_PAL.length)]);
    state.peds.push({ id: i, edge: e, s: rng() * (e.len || 1), speed: 1.1 + rng() * 0.8, color: col });
  }
}

// pick the next lane at a node; straight preferred, turns allowed
function pickNext(agent) {
  const nodeId = agent.edge.to;
  const n = state.nodes.get(nodeId);
  if (!n || !n.out.length) { agent.s = 0; return; }
  const dinx = agent.edge.dx, dinz = agent.edge.dz;
  let best = -1, bd = -2;
  for (const idx of n.out) {
    const e = state.edges[idx];
    const d = dinx * e.dx + dinz * e.dz;
    if (d > bd) { bd = d; best = idx; }
  }
  const r = randFrom((agent.id * 7919 ^ nodeId) >>> 0);
  let choice;
  if (bd > 0.90 || r < 0.55) {
    choice = best;                                  // keep going straight
  } else {
    const alt = n.out.filter(idx => idx !== best);
    if (alt.length) choice = alt[Math.floor(r * alt.length)];
    else choice = best;
  }
  agent.edge = state.edges[choice];
  agent.s = 0;
}

// ---------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------
function buildMeshes(world, maxCars, maxPeds) {
  const scene = world.scene;

  if (state.carMesh) { scene.remove(state.carMesh); state.carMesh.geometry.dispose(); }
  if (state.lightMesh) { scene.remove(state.lightMesh); state.lightMesh.geometry.dispose(); }
  if (state.pedMesh) { scene.remove(state.pedMesh); state.pedMesh.geometry.dispose(); }

  const carGeo = carGeometry();
  const carMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3, metalness: 0.62 });
  const cm = new THREE.InstancedMesh(carGeo, carMat, maxCars);
  cm.frustumCulled = false;
  cm.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  cm.count = 0;
  scene.add(cm); state.carMesh = cm;

  const lightGeo = lightGeometry();
  const lightMat = new THREE.MeshStandardMaterial({
    color: 0x000000, emissive: 0xffffff, emissiveIntensity: 1.4,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false,
  });
  const lm = new THREE.InstancedMesh(lightGeo, lightMat, maxCars * 2);
  lm.frustumCulled = false;
  lm.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  lm.count = 0;
  scene.add(lm); state.lightMesh = lm;

  const pedGeo = pedGeometry();
  const pedMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 });
  const pm = new THREE.InstancedMesh(pedGeo, pedMat, maxPeds);
  pm.frustumCulled = false;
  pm.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  pm.count = 0;
  scene.add(pm); state.pedMesh = pm;
}

// ---------------------------------------------------------------------------
// Night factor from time of day (headlights ramp up at dusk)
// ---------------------------------------------------------------------------
function nightFactor(sec) {
  const h = sec / 3600;
  if (h >= 7 && h <= 18) return 0;
  if (h < 5 || h > 20.5) return 1;
  const ramp = (x, a, b) => clamp((x - a) / (b - a), 0, 1);
  const rise = ramp(h, 5, 7);          // dawn -> 1
  const set = 1 - ramp(h, 18, 20.5);   // dusk -> 1
  return clamp(1 - Math.max(rise, set) * 0.9, 0, 1);
}

function compose(mesh, idx, x, y, z, ry, color) {
  _pos.set(x, y, z);
  _eul.set(0, ry, 0);
  _quat.setFromEuler(_eul);
  _m4.compose(_pos, _quat, _one);
  mesh.setMatrixAt(idx, _m4);
  if (color) mesh.setColorAt(idx, color);
}

// ---------------------------------------------------------------------------
// Per-frame update
// ---------------------------------------------------------------------------
function step(world, dtSec) {
  const cam = world && world.camera;
  // advance cars
  for (const c of state.cars) {
    c.s += c.speed * dtSec;
    if (c.s >= c.edge.len) { const over = c.s - c.edge.len; pickNext(c); c.s = Math.min(over, 1); }
  }
  for (const p of state.peds) {
    p.s += p.speed * dtSec;
    if (p.s >= p.edge.len) { const over = p.s - p.edge.len; pickNext(p); p.s = Math.min(over, 1); }
  }

  // night headlight intensity
  const nf = world && world.meta ? nightFactor(world.meta.timeOfDaySec) : 0;
  if (state.lightMesh) state.lightMesh.material.emissiveIntensity = 0.15 + nf * 1.5;

  // ----- cars -----
  const carMesh = state.carMesh, lightMesh = state.lightMesh;
  if (carMesh && cam) {
    const cx = cam.position.x, cz = cam.position.z;
    const order = [];
    for (let i = 0; i < state.cars.length; i++) {
      const c = state.cars[i];
      const p = edgeCarPos(c.edge, clamp(c.s / (c.edge.len || 1), 0, 1));
      const y = heightAt(world, p.x, p.z);
      c.px = p.x; c.py = y + 0.18; c.pz = p.z;
      const d2 = (p.x - cx) * (p.x - cx) + (p.z - cz) * (p.z - cz);
      order.push([i, d2]);
    }
    order.sort((a, b) => a[1] - b[1]);
    let vis = 0;
    const cut2 = CAR_CULL * CAR_CULL;
    while (vis < order.length && order[vis][1] <= cut2) vis++;
    carMesh.count = vis;
    for (let k = 0; k < vis; k++) {
      const c = state.cars[order[k][0]];
      const ry = Math.atan2(c.edge.dx, c.edge.dz);
      compose(carMesh, k, c.px, c.py, c.pz, ry, c.color);
      // headlight (front, white) and taillight (rear, red), placed by CAR length
      const hl = 2.1, tl = 2.3;
      _pos.set(c.px + c.edge.dx * hl, c.py + 0.32, c.pz + c.edge.dz * hl);
      _eul.set(0, ry, 0); _quat.setFromEuler(_eul); _m4.compose(_pos, _quat, _one);
      lightMesh.setMatrixAt(k * 2, _m4);
      lightMesh.setColorAt(k * 2, HEADLIGHT_COLOR);
      _pos.set(c.px - c.edge.dx * tl, c.py + 0.32, c.pz - c.edge.dz * tl);
      _m4.compose(_pos, _quat, _one);
      lightMesh.setMatrixAt(k * 2 + 1, _m4);
      lightMesh.setColorAt(k * 2 + 1, TAILLIGHT_COLOR);
    }
    carMesh.instanceMatrix.needsUpdate = true;
    if (carMesh.instanceColor) carMesh.instanceColor.needsUpdate = true;
    lightMesh.count = vis * 2;
    lightMesh.instanceMatrix.needsUpdate = true;
    if (lightMesh.instanceColor) lightMesh.instanceColor.needsUpdate = true;
  }

  // ----- peds -----
  const pedMesh = state.pedMesh;
  if (pedMesh && cam) {
    const cx = cam.position.x, cz = cam.position.z;
    const order = [];
    for (let i = 0; i < state.peds.length; i++) {
      const p = state.peds[i];
      const q = edgePedPos(p.edge, clamp(p.s / (p.edge.len || 1), 0, 1));
      const y = heightAt(world, q.x, q.z);
      p.px = q.x; p.py = y; p.pz = q.z;
      const d2 = (q.x - cx) * (q.x - cx) + (q.z - cz) * (q.z - cz);
      order.push([i, d2]);
    }
    order.sort((a, b) => a[1] - b[1]);
    let vis = 0;
    const cut2 = PED_CULL * PED_CULL;
    while (vis < order.length && order[vis][1] <= cut2) vis++;
    pedMesh.count = vis;
    for (let k = 0; k < vis; k++) {
      const p = state.peds[order[k][0]];
      compose(pedMesh, k, p.px, p.py, p.pz, Math.atan2(p.edge.dx, p.edge.dz), p.color);
    }
    pedMesh.instanceMatrix.needsUpdate = true;
    if (pedMesh.instanceColor) pedMesh.instanceColor.needsUpdate = true;
  }

  // mirror to world.agents for other systems (audio scales with this length)
  const all = [];
  for (const c of state.cars) {
    all.push({ type: 'car', x: c.px !== undefined ? c.px : c.edge.cx0, y: c.py !== undefined ? c.py : 0, z: c.pz !== undefined ? c.pz : c.edge.cz0, speed: c.speed, dir: { x: c.edge.dx, z: c.edge.dz } });
  }
  for (const p of state.peds) {
    all.push({ type: 'ped', x: p.px !== undefined ? p.px : p.edge.cx0, y: p.py !== undefined ? p.py : 0, z: p.pz !== undefined ? p.pz : p.edge.cz0, speed: p.speed, dir: { x: p.edge.dx, z: p.edge.dz } });
  }
  world.agents = all;
}

// ---------------------------------------------------------------------------
// Rebuild the network + spawn agents (on init and when roads change)
// ---------------------------------------------------------------------------
function rebuild(world) {
  const seed = ((world.meta && world.meta.seed) || 1337) ^ 0x51ab3d5f >>> 0;
  const rng = mulberry32(seed);

  let segs = null;
  state.isDemo = false;
  if (Array.isArray(world.roads) && world.roads.length > 0) {
    segs = world.roads.map(s => ({
      ax: s.from[0], az: s.from[1], bx: s.to[0], bz: s.to[1],
      width: s.width || 14,
    }));
  }
  if (!segs || !segs.length) {
    // no drivable network yet -> small ring-road loop so traffic is visible
    state.isDemo = true;
    segs = [];
    const R = 75, CX = 0, CZ = -30, N = 30;
    for (let i = 0; i < N; i++) {
      const a0 = (i / N) * Math.PI * 2, b0 = ((i + 1) / N) * Math.PI * 2;
      segs.push({
        ax: CX + R * Math.cos(a0), az: CZ + R * Math.sin(a0),
        bx: CX + R * Math.cos(b0), bz: CZ + R * Math.sin(b0),
        width: 14,
      });
    }
  }

  buildGraph(segs);
  spawnCars(rng);
  spawnPeds(rng);
  buildMeshes(world, state.cars.length || 8, state.peds.length || 4);

  state.edgeCountCached = Array.isArray(world.roads) ? world.roads.length : 0;
  state.built = true;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
export function init(world) {
  // if roads change later, keep our graph in sync
  bus.on('road:add', () => { try { rebuild(world); } catch (e) { /* isolation */ } });
  bus.on('road:remove', () => { try { rebuild(world); } catch (e) { /* isolation */ } });
  try { rebuild(world); } catch (e) { console.error('[traffic] init failed', e); }
}

function logErr(where, e) {
  if (!globalThis.__trafficLastErr) {
    globalThis.__trafficLastErr = where + ': ' + String(e && (e.message || e));
  }
}

export function update(dtSec, world) {
  if (!state.built || !world || !world.scene) return;
  // detect a road network that appeared/changed and rebuild once
  const rc = Array.isArray(world.roads) ? world.roads.length : 0;
  if (rc !== state.edgeCountCached) {
    try { rebuild(world); } catch (e) { logErr('rebuild', e); }
  }
  try { step(world, dtSec || 0.05); } catch (e) { logErr('step', e); }
}

export function showcase(container) {
  const mode = state.isDemo ? 'loop demo (no road network yet)' : `${state.edges.length} lanes`;
  if (container) {
    container.innerHTML = `<div style="padding:12px;font-family:sans-serif;color:#cfd6d9;
      background:#0a1117">traffic — ${state.cars.length} cars, ${state.peds.length} pedestrians
      · ${mode} · headlights/taillights at night</div>`;
  }
}

// module-scope light colours (allocated once)
const HEADLIGHT_COLOR = new THREE.Color(1.0, 0.96, 0.86);
const TAILLIGHT_COLOR = new THREE.Color(1.0, 0.18, 0.10);
