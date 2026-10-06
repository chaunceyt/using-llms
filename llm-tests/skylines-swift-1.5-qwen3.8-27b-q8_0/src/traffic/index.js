// traffic — believable vehicle flow + pedestrian life on the road graph
// (CS2 money shot: head/taillights at night, people on the sidewalks by day).
//
// Draw calls: 7 total — one InstancedMesh per vehicle kind (car / truck / bus),
// one InstancedMesh of emissive light quads (2 head + 2 tail per vehicle), one
// additive InstancedMesh of headlight-beam pools, and two InstancedMeshes for
// pedestrians (capsule bodies with per-instance clothing colour + head spheres).
//
// Pedestrians: ~850 figures walk the sidewalk bands of the SAME routing graph
// the vehicles use (right-hand side of travel), cross the street at
// intersections (crosswalks), cut corners, and linger on corners near
// commercial zones. Spawn weights + linger chance come from the zoning grid
// (commercial density), and the active count follows a day curve (peak noon +
// evening, sparse at deep night). One InstancedMesh per part = 2 draw calls.
//
// Determinism: every random value comes from ctx.rng.fork('traffic').
// Performance: O(fleet + peds) per frame with tiny constants (~700 vehicles,
// ~850 peds), no per-frame allocations (module-scope scratch only).
//
// Graph note: the roads module's grid lines wobble, so edge endpoints do NOT
// coincide at interior crossings. We therefore rebuild the routing graph here:
// detect centerline crossings between all edge pairs, split edges at each
// crossing, dedupe nodes, and precompute per-edge-end turn transitions
// (straight preferred, U-turn last, dead ends respawn).
//
// Public API: setDensity(d), spawn(kind), world.traffic = { vehicles, active }.
// Emits 'traffic:ready' {vehicles}, 'traffic:density' {d}.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const smooth01 = (x) => { const t = clamp(x, 0, 1); return t * t * (3 - 2 * t); };

// 2D segment intersection in XZ. Returns {x, z, t, u} (t along seg 1, u along seg 2).
function segInt(ax, az, bx, bz, cx, cz, dx, dz) {
  const r1x = bx - ax, r1z = bz - az, r2x = dx - cx, r2z = dz - cz;
  const den = r1x * r2z - r1z * r2x;
  if (Math.abs(den) < 1e-12) return null;
  const t = ((cx - ax) * r2z - (cz - az) * r2x) / den;
  const u = ((cx - ax) * r1z - (cz - az) * r1x) / den;
  if (t < 1e-6 || t > 1 - 1e-6 || u < 1e-6 || u > 1 - 1e-6) return null;
  return { x: ax + t * r1x, z: az + t * r1z, t, u };
}

// ---- geometry helpers (massing: tapered boxes + wheel cylinders) -------------
const V_WHITE = [1, 1, 1];
const V_GLASS = [0.10, 0.12, 0.16];
const V_WHEEL = [0.045, 0.045, 0.055];

function _tint(g, c) {
  const n = g.attributes.position.count;
  const arr = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { arr[i * 3] = c[0]; arr[i * 3 + 1] = c[1]; arr[i * 3 + 2] = c[2]; }
  g.setAttribute('color', new THREE.BufferAttribute(arr, 3));
}

// Box whose top face is inset (frustum) — reads as car body/cabin massing.
// Origin at ground (y=0 is the wheel line), +Z is forward.
function _box(w, h, d, inset, y0, color, xoff = 0, zoff = 0) {
  const g = new THREE.BoxGeometry(w, h, d);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const y = p.getY(i);
    if (y > 0) { p.setX(i, p.getX(i) * inset); p.setZ(i, p.getZ(i) * inset); }
    p.setY(i, y + h * 0.5 + y0);
  }
  g.translate(xoff, 0, zoff);
  g.computeVertexNormals();
  _tint(g, color);
  return g;
}

function _wheel(r, w, x, y, z) {
  const g = new THREE.CylinderGeometry(r, r, w, 10);
  g.rotateZ(Math.PI / 2);
  g.translate(x, y, z);
  _tint(g, V_WHEEL);
  return g;
}

// CAR — two-mass profile (low body slab + inset greenhouse) with a distinct
// hood, trunk deck and a rounded roofline, so it reads as a car at street AND
// aerial scale. A dark rocker band fakes the wheel-arch shadows.
function _carGeo() {
  const g = mergeGeometries([
    _box(1.82, 0.58, 4.42, 0.98, 0.16, V_WHITE),            // lower body slab
    _box(1.80, 0.30, 1.30, 0.95, 0.44, V_WHITE, 0, 1.44),   // hood (front deck)
    _box(1.80, 0.28, 1.04, 0.95, 0.44, V_WHITE, 0, -1.58),  // trunk deck (rear)
    _box(1.58, 0.52, 2.02, 0.58, 0.66, V_GLASS, 0, -0.20),  // greenhouse (inset roof)
    _box(1.38, 0.10, 1.56, 0.74, 1.12, V_WHITE, 0, -0.20),  // roofline cap
    _box(1.90, 0.16, 3.44, 1.0, 0.10, V_WHEEL),             // rocker / arch shadow band
    _wheel(0.34, 0.30, -0.78, 0.34, 1.38),
    _wheel(0.34, 0.30, 0.78, 0.34, 1.38),
    _wheel(0.34, 0.30, -0.78, 0.34, -1.38),
    _wheel(0.34, 0.30, 0.78, 0.34, -1.38),
  ]);
  g.scale(1.12, 1.06, 1.12);
  return g;
}

// TRUCK — low chassis + tall cab (with glass) + big box trailer; twin rear axles.
function _truckGeo() {
  const g = mergeGeometries([
    _box(2.34, 0.50, 8.20, 1.0, 0.16, V_WHITE),              // chassis
    _box(2.30, 2.10, 2.20, 0.92, 0.55, V_WHITE, 0, 2.85),    // cab
    _box(2.10, 0.82, 1.10, 0.82, 1.12, V_GLASS, 0, 3.16),    // cab windshield band
    _box(2.50, 2.60, 5.40, 1.0, 0.55, V_WHITE, 0, -1.15),    // box trailer
    _box(2.56, 0.18, 5.40, 1.0, 0.10, V_WHEEL, 0, -1.15),    // trailer lower shadow
    _wheel(0.48, 0.34, -1.05, 0.48, 2.85),
    _wheel(0.48, 0.34, 1.05, 0.48, 2.85),
    _wheel(0.48, 0.34, -1.05, 0.48, -0.55),
    _wheel(0.48, 0.34, 1.05, 0.48, -0.55),
    _wheel(0.48, 0.34, -1.05, 0.48, -2.30),
    _wheel(0.48, 0.34, 1.05, 0.48, -2.30),
  ]);
  g.scale(1.12, 1.06, 1.12);
  return g;
}

// BUS — long box with a full-length window band and a roofline; twin axles.
function _busGeo() {
  const g = mergeGeometries([
    _box(2.55, 2.70, 11.5, 0.99, 0.30, V_WHITE),            // body
    _box(2.59, 0.72, 10.2, 0.995, 1.50, V_GLASS),           // window band (proud: no z-fight)
    _box(2.40, 0.14, 11.3, 0.98, 2.62, V_WHITE),            // roofline cap
    _box(2.60, 0.20, 11.3, 1.0, 0.10, V_WHEEL),             // lower shadow band
    _wheel(0.55, 0.36, -1.10, 0.55, 3.9),
    _wheel(0.55, 0.36, 1.10, 0.55, 3.9),
    _wheel(0.55, 0.36, -1.10, 0.55, -3.9),
    _wheel(0.55, 0.36, 1.10, 0.55, -3.9),
  ]);
  g.scale(1.12, 1.06, 1.12);
  return g;
}

// Light quad offsets per kind (local space, +Z forward): head (±hx, hy, +hz), tail (±tx, ty, -tz).
// Scaled to the 1.12x (x/z) / 1.06x (y) bulked geometry.
const LIGHT_OFF = [
  { hx: 0.67, hy: 0.55, hz: 2.52, tx: 0.67, ty: 0.64, tz: 2.52 },  // car
  { hx: 0.95, hy: 1.11, hz: 4.46, tx: 1.12, ty: 1.80, tz: 4.35 },  // truck (taillights high on trailer)
  { hx: 1.06, hy: 0.95, hz: 6.48, tx: 1.12, ty: 1.64, tz: 6.48 },  // bus
];

// ---- headlight beam (night only) --------------------------------------------
// A flat trapezoid pool on the road in front of the headlights (warm, fading to
// nothing) merged with a faint red pool behind the taillights. One instance per
// vehicle; the whole mesh is additive and ramped to 0 by day. Per-vertex colour
// gives the near->far falloff; the instance colour scales overall intensity.
function _beamQuad(y, xn0, zn0, xn1, zn1, xf2, zf2, xf3, zf3, cNear, cFar) {
  const pos = new Float32Array([
    xn0, y, zn0,  xn1, y, zn1,  xf2, y, zf2,   // tri 1
    xn0, y, zn0,  xf2, y, zf2,  xf3, y, zf3,   // tri 2
  ]);
  const col = new Float32Array([
    cNear[0], cNear[1], cNear[2],
    cNear[0], cNear[1], cNear[2],
    cFar[0],  cFar[1],  cFar[2],
    cNear[0], cNear[1], cNear[2],
    cFar[0],  cFar[1],  cFar[2],
    cFar[0],  cFar[1],  cFar[2],
  ]);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return g;
}

function _beamGeo() {
  // y = 0.16: clears the zebra crosswalks (which sit +0.10 above the asphalt)
  // so the pool never vanishes under a crossing at an intersection.
  const head = _beamQuad(0.16, -0.78, 2.30, 0.78, 2.30, 3.15, 12.5, -3.15, 12.5,
    [1.30, 1.18, 0.98], [0.0, 0.0, 0.0]);            // headlight pool, forward
  const tail = _beamQuad(0.16, -1.15, -2.30, 1.15, -2.30, 0.95, -5.8, -0.95, -5.8,
    [1.10, 0.11, 0.06], [0.0, 0.0, 0.0]);            // tail glow, behind
  return mergeGeometries([head, tail]);
}

// ---- pedestrian geometry -----------------------------------------------------
// A person = capsule body + head sphere (two InstancedMeshes so the head keeps
// a fixed skin tone while the body takes per-instance clothing colour).
// ~1.36 m tall before the 0.92-1.10 per-instance height scale.
function _pedBodyGeo() {
  const g = new THREE.CapsuleGeometry(0.17, 0.60, 3, 8);
  g.translate(0, 0.56, 0); // spans y 0.09..1.03
  return g;
}
function _pedHeadGeo() {
  const g = new THREE.SphereGeometry(0.115, 8, 6);
  g.translate(0, 1.24, 0); // top ~1.355
  return g;
}

// Believable paint palettes (sRGB hex, weighted). Light cars dominate so the
// fleet pops against dark asphalt at street AND aerial scale; a few saturated
// accents (red/blue/taxi) for life. Buses transit-blue.
const CAR_PAINTS = [
  ['#f2f4f6', 14], ['#e4e7ea', 12],         // bright whites
  ['#c3c8d0', 11], ['#aeb2ba', 9],          // silvers
  ['#82868e', 6], ['#565a61', 4],           // greys
  ['#202127', 5], ['#111216', 3],           // blacks
  ['#2c4a78', 7], ['#213353', 5],           // blues
  ['#9a3531', 6], ['#b04438', 5],           // reds
  ['#e0bc30', 7],                           // taxis
  ['#3a5a42', 3], ['#5f4c34', 2], ['#2f6260', 2],
];
const TRUCK_PAINTS = [
  ['#f2f4f6', 22], ['#c3c8d0', 14], ['#82868e', 10],
  ['#2c4a78', 9], ['#565a61', 7], ['#9a3531', 5], ['#c9601f', 4],
];
const BUS_PAINTS = [
  ['#1d4f91', 90], ['#c3c8d0', 10],
];

// ---- pedestrians -------------------------------------------------------------
const PED_TOTAL = 850;                 // capped instanced figures
const PED_LAT = [5.7, 6.3];            // sidewalk band is 5.0-6.5 m from centerline

// Everyday street-clothes palette (sRGB hex, weighted). Mid-to-high albedo so
// figures read at street distance in day AND night; a spread of hues for life.
const PED_CLOTHES = [
  ['#e8e4da', 6], ['#d8d2c4', 5],         // cream / sand
  ['#c9ccd2', 5], ['#b9bdc4', 6],         // light greys
  ['#8a8f97', 7], ['#5b6470', 8],         // slate
  ['#4a5568', 7], ['#31424e', 4],         // denim / petrol
  ['#2c2e33', 6],                         // charcoal
  ['#7d3b34', 5], ['#93453a', 4],         // brick / rust
  ['#3c5a6e', 5], ['#2f4a56', 4],         // teal
  ['#5d5a3a', 4], ['#7a6a45', 4],         // olive / tan
  ['#54404f', 3], ['#6b4a5e', 3],         // plum
];

export default class Traffic {
  name = 'traffic';

  constructor() {
    this._root = null;
    this._geos = [];
    this._bodyMesh = [];
    this._lightMesh = null;
    this._beamMesh = null;
    this._bodyMat = null;
    this._lightMat = null;
    this._beamMat = null;
    this._vehicles = [];
    this._edges = [];        // { pts, len, nodeA, nodeB, aUnit, bUnit }
    this._nodes = [];
    this._trans = [];        // [edge*2 + arrivalEnd] -> [{e, dir, score}]
    this._edgeCum = null;    // cumulative edge lengths (weighted spawns)
    this._edgeTotal = 0;
    this._count = [0, 0, 0];
    this._caps = [620, 78, 58];
    this._fleetMix = [580, 72, 50]; // cars ~83%, trucks ~10%, buses ~7%  (~702 total)
    this._lightCursor = 0;
    this._userDensity = 1;
    this._rng = null;
    this._terrainMod = null;
    this._ctx = null;
    // pedestrians
    this._peds = [];
    this._pedBodyMesh = null;
    this._pedHeadMesh = null;
    this._pedMat = null;
    this._pedHeadMat = null;
    this._pedCum = null;
    this._pedTotal = 0;
    this._nodeComm = [];
    // showcase staging
    this._showScene = null;
    this._showExtras = [];
    this._showSun = null;
    this._showHemi = null;
    this._terrainRoot = null;
    this._roadsRoot = null;
  }

  async init(world, ctx) {
    this._ctx = ctx;
    this._rng = ctx.rng.fork('traffic');
    this._terrainMod = ctx.registry ? ctx.registry.get('terrain') : null;
    const roads = ctx.registry ? ctx.registry.get('roads') : null;

    this._buildGraph(roads ? roads.graph.edges : []);
    if (this._edges.length) this._buildPedWeights();
    this._buildGeos();
    this._buildMeshes();
    if (this._edges.length) this._buildFleet();

    world.traffic = {
      vehicles: this._vehicles, active: 0,
      peds: this._peds.length, pedsActive: 0,
    };
    ctx.scene.add(this._root);
    ctx.events.emit('traffic:ready', { vehicles: this._vehicles.length });
    if (this._vehicles.length) this.update(0, world);
  }

  // ---- routing graph ---------------------------------------------------------
  _buildGraph(srcEdges) {
    const tmod = this._terrainMod;
    const getH = (x, z) => (tmod && typeof tmod.getHeight === 'function') ? tmod.getHeight(x, z) : 0;

    // normalize source edges (copy pts so we never mutate roads' data)
    const edges = srcEdges.map((e) => ({
      pts: e.pts.map((p) => ({ x: p.x, z: p.z, y: (p.y != null ? p.y : getH(p.x, p.z) + 0.25), _s: p._s || 0 })),
    }));

    // 1) find every centerline crossing between edge pairs
    const hits = new Map(); // edge index -> [{f, x, z}]
    for (let i = 0; i < edges.length; i++) {
      const P1 = edges[i].pts;
      for (let j = i + 1; j < edges.length; j++) {
        const P2 = edges[j].pts;
        const n1 = P1.length - 1, n2 = P2.length - 1;
        for (let a = 0; a < n1; a++) {
          for (let b = 0; b < n2; b++) {
            const r = segInt(P1[a].x, P1[a].z, P1[a + 1].x, P1[a + 1].z, P2[b].x, P2[b].z, P2[b + 1].x, P2[b + 1].z);
            if (!r) continue;
            let h1 = hits.get(i); if (!h1) { h1 = []; hits.set(i, h1); }
            let h2 = hits.get(j); if (!h2) { h2 = []; hits.set(j, h2); }
            h1.push({ f: (a + r.t) / n1, x: r.x, z: r.z });
            h2.push({ f: (b + r.u) / n2, x: r.x, z: r.z });
          }
        }
      }
    }

    // 2) split each edge at its (deduped, sorted) crossings
    const sub = [];
    for (let i = 0; i < edges.length; i++) {
      const pts = edges[i].pts;
      const n = pts.length - 1;
      let totalM = 0;
      for (let m = 1; m < pts.length; m++) totalM += Math.hypot(pts[m].x - pts[m - 1].x, pts[m].z - pts[m - 1].z);
      const hs = (hits.get(i) || []).slice().sort((p, q) => p.f - q.f);
      const kept = [];
      let lastM = -10;
      for (const h of hs) {
        const sM = h.f * totalM;
        if (sM < 2.5 || totalM - sM < 2.5) continue; // too close to an endpoint
        if (sM - lastM < 4) continue;                // duplicate / too close to previous
        kept.push(h); lastM = sM;
      }
      const bounds = [0, ...kept.map((h) => h.f), 1];
      for (let k = 0; k < bounds.length - 1; k++) {
        const pts2 = this._slicePts(pts, bounds[k], bounds[k + 1]);
        if (pts2.length < 2) continue;
        let len = 0;
        for (let m = 1; m < pts2.length; m++) len += Math.hypot(pts2[m].x - pts2[m - 1].x, pts2[m].z - pts2[m - 1].z);
        if (len < 2.5) continue;
        sub.push({ pts: pts2, len });
      }
    }

    // 2b) Bridge dead-end endpoints to the nearest nearby edge.
    // Road centerlines wobble, so at many junctions (especially near map edges)
    // the two centerlines MISS each other by a few metres even though the 10 m
    // wide asphalt ribbons overlap continuously. Stitch such gaps with short
    // connector edges so the whole network stays routable (no long dead ends).
    {
      const keyOf = (p) => Math.round(p.x * 2) + ',' + Math.round(p.z * 2);
      const arcLen = (pts) => {
        let L = 0;
        for (let m = 1; m < pts.length; m++) L += Math.hypot(pts[m].x - pts[m - 1].x, pts[m].z - pts[m - 1].z);
        return L;
      };
      let iter = 0;
      while (iter++ < 80) {
        // first un-bridged dead-end endpoint
        const count = new Map();
        for (const e of sub) {
          const ka = keyOf(e.pts[0]), kb = keyOf(e.pts[e.pts.length - 1]);
          count.set(ka, (count.get(ka) || 0) + 1);
          count.set(kb, (count.get(kb) || 0) + 1);
        }
        let found = null;
        for (let i = 0; i < sub.length && !found; i++) {
          const e = sub[i];
          if (e.connector) continue;
          const pa = e.pts[0], pb = e.pts[e.pts.length - 1];
          if (!e._noBridgeA && count.get(keyOf(pa)) === 1) found = { e, P: pa };
          else if (!e._noBridgeB && count.get(keyOf(pb)) === 1) found = { e, P: pb };
        }
        if (!found) break;
        const P = found.P;
        const feIdx = sub.indexOf(found.e);
        // nearest point on any other non-connector edge (not sharing P's node)
        let best = null;
        for (let k = 0; k < sub.length; k++) {
          const e2 = sub[k];
          if (k === feIdx || e2.connector) continue;
          const q = e2.pts;
          if (keyOf(P) === keyOf(q[0]) || keyOf(P) === keyOf(q[q.length - 1])) continue;
          const nq = q.length - 1;
          for (let s = 0; s < nq; s++) {
            const dx = q[s + 1].x - q[s].x, dz = q[s + 1].z - q[s].z;
            const l2 = dx * dx + dz * dz;
            let u = l2 > 0 ? ((P.x - q[s].x) * dx + (P.z - q[s].z) * dz) / l2 : 0;
            u = u < 0 ? 0 : u > 1 ? 1 : u;
            const cx = q[s].x + u * dx, cz = q[s].z + u * dz;
            const d = Math.hypot(P.x - cx, P.z - cz);
            if (!best || d < best.d) best = { k, f: (s + u) / nq, d, cx, cz };
          }
        }
        if (!best || best.d < 0.5 || best.d > 8) {
          // cannot bridge (map boundary); remember so we stop retrying
          const e = found.e;
          if (e.pts[0] === P) e._noBridgeA = true; else e._noBridgeB = true;
          continue;
        }
        // split the target edge at the attachment point (snap if it would create a < 2.5 m stub)
        const e2 = sub[best.k];
        const q = e2.pts;
        const L2 = arcLen(q);
        let f = best.f;
        if (f * L2 < 2.5) f = 0;
        else if (L2 - f * L2 < 2.5) f = 1;
        let Np = null;
        if (f > 0 && f < 1) {
          const left = this._slicePts(q, 0, f);
          const right = this._slicePts(q, f, 1);
          const ll = arcLen(left), rl = arcLen(right);
          if (left.length >= 2 && right.length >= 2 && ll >= 2.5 && rl >= 2.5) {
            sub.splice(best.k, 1, { pts: left, len: ll }, { pts: right, len: rl });
            Np = right[0];
          } else {
            f = 0;
          }
        }
        if (!Np) Np = q[0];
        // the connector edge (lies on the overlapping asphalt at the junction)
        const mid = { x: (P.x + Np.x) / 2, z: (P.z + Np.z) / 2, y: (P.y + Np.y) / 2, _s: 0 };
        sub.push({ pts: [P, mid, Np], len: Math.hypot(P.x - Np.x, P.z - Np.z), connector: true });
      }
    }

    // 3) nodes (0.5 m dedupe) + incident edges
    const nodeMap = new Map();
    const nodes = [];
    const nodeOf = (p) => {
      const key = Math.round(p.x * 2) + ',' + Math.round(p.z * 2);
      let id = nodeMap.get(key);
      if (id == null) { id = nodes.length; nodes.push({ x: p.x, z: p.z, incident: [] }); nodeMap.set(key, id); }
      return id;
    };
    for (let i = 0; i < sub.length; i++) {
      const e = sub[i];
      e.nodeA = nodeOf(e.pts[0]);
      e.nodeB = nodeOf(e.pts[e.pts.length - 1]);
      const a = e.pts[0], a2 = e.pts[1];
      let l = Math.hypot(a2.x - a.x, a2.z - a.z) || 1;
      e.aUnit = { x: (a2.x - a.x) / l, z: (a2.z - a.z) / l };
      const b = e.pts[e.pts.length - 1], b0 = e.pts[e.pts.length - 2];
      l = Math.hypot(b.x - b0.x, b.z - b0.z) || 1;
      e.bUnit = { x: (b.x - b0.x) / l, z: (b.z - b0.z) / l };
      nodes[e.nodeA].incident.push({ e: i, end: 0 });
      nodes[e.nodeB].incident.push({ e: i, end: 1 });
    }

    // 4) transitions: for each edge + arrival end, candidate next edges ranked by
    //    turn angle (straight first, U-turn last). Ties broken by a deterministic
    //    pre-shuffle so "random among the rest" costs nothing at runtime.
    const trans = [];
    for (let i = 0; i < sub.length; i++) {
      for (let end = 0; end < 2; end++) { // 0 = arrived at a (dir -1), 1 = arrived at b (dir +1)
        const e = sub[i];
        const inU = end === 1 ? e.bUnit : { x: -e.aUnit.x, z: -e.aUnit.z };
        const node = end === 1 ? nodes[e.nodeB] : nodes[e.nodeA];
        const cands = [];
        for (const inc of node.incident) {
          if (inc.e === i) continue;
          const f = sub[inc.e];
          const dir = inc.end === 0 ? 1 : -1;
          const outU = dir === 1 ? f.aUnit : { x: -f.bUnit.x, z: -f.bUnit.z };
          cands.push({ e: inc.e, dir, score: inU.x * outU.x + inU.z * outU.z });
        }
        for (let k = cands.length - 1; k > 0; k--) {
          const j = (this._rng.next() * (k + 1)) | 0;
          const tmp = cands[k]; cands[k] = cands[j]; cands[j] = tmp;
        }
        cands.sort((p, q) => q.score - p.score);
        // dead end: no forward option -> U-turn (cul-de-sac) instead of respawning
        if (!cands.length) cands.push({ e: i, dir: end === 1 ? -1 : 1, score: -1 });
        trans.push(cands);
      }
    }

    // weighted edge picker (proportional to length)
    let total = 0;
    const cum = [];
    for (const e of sub) { total += e.len; cum.push(total); }
    this._edges = sub;
    this._nodes = nodes;
    this._trans = trans;
    this._edgeCum = cum;
    this._edgeTotal = total;
  }

  // pts between fractional indices fa..fb (inclusive ends, interpolated).
  _slicePts(pts, fa, fb) {
    const n = pts.length - 1;
    const i0 = Math.ceil(fa * n - 1e-6);
    const i1 = Math.floor(fb * n + 1e-6);
    const out = [];
    const interp = (ff) => {
      const g = ff * n;
      const i = Math.min(n - 1, g | 0);
      const fr = g - i;
      const A = pts[i], B = pts[i + 1];
      return { x: A.x + (B.x - A.x) * fr, z: A.z + (B.z - A.z) * fr, y: A.y + (B.y - A.y) * fr, _s: 0 };
    };
    if (fa > 0) out.push(interp(fa));
    for (let i = i0; i <= i1; i++) out.push(pts[i]);
    if (fb < 1) out.push(interp(fb));
    return out;
  }

  _pickEdge() {
    const r = this._rng.next() * this._edgeTotal;
    const cum = this._edgeCum;
    for (let i = 0; i < cum.length; i++) if (r <= cum[i]) return i;
    return cum.length - 1;
  }

  _pickPaint(entries) {
    let tot = 0;
    for (const e of entries) tot += e[1];
    let r = this._rng.next() * tot;
    for (const e of entries) { r -= e[1]; if (r <= 0) return e[0]; }
    return entries[entries.length - 1][0];
  }

  // ---- pedestrian routing weights ----------------------------------------------
  // Per-edge commercial presence (zoning density sampled along the centerline)
  // drives spawn weights — sidewalks by the commercial core run 2-3x denser —
  // and per-node commercial scores drive corner lingering.
  _buildPedWeights() {
    const grid = this._ctx.world.zoning.grid; // Map "tx,tz" -> {x,z,use,density}
    const tileAt = (x, z) => {
      const tx = Math.floor((x + 140) / 8), tz = Math.floor((z + 140) / 8);
      if (tx < 0 || tx >= 35 || tz < 0 || tz >= 35) return null;
      return grid.get(tx + ',' + tz) || null;
    };
    const commAt = (x, z) => {
      let s = 0;
      for (let k = 0; k < 5; k++) {
        const ox = k === 0 ? 0 : k === 1 ? 7 : k === 2 ? -7 : 0;
        const oz = k === 0 ? 0 : k < 3 ? 0 : k === 3 ? 7 : -7;
        const t = tileAt(x + ox, z + oz);
        if (t && t.use === 'commercial') s += t.density;
      }
      return s;
    };

    let maxF = 0;
    for (const e of this._edges) {
      const pts = e.pts;
      let s = 0;
      for (let i = 0; i < pts.length; i++) s += commAt(pts[i].x, pts[i].z);
      e._pc = pts.length ? s / pts.length : 0;
      if (e._pc > maxF) maxF = e._pc;
    }
    let tot = 0;
    const cum = [];
    for (const e of this._edges) {
      const f = maxF > 0 ? e._pc / maxF : 0;
      // connector stubs cut across junctions — not real sidewalks; don't spawn there
      const w = e.connector ? 0 : e.len * (0.30 + 0.70 * f);
      tot += w;
      cum.push(tot);
    }
    this._pedCum = cum;
    this._pedTotal = tot;
    this._nodeComm = this._nodes.map((nd) => Math.min(1, commAt(nd.x, nd.z) / 6));
  }

  _pickPedEdge() {
    if (this._pedTotal <= 0) return (this._rng.next() * this._edges.length) | 0;
    const r = this._rng.next() * this._pedTotal;
    const cum = this._pedCum;
    for (let i = 0; i < cum.length; i++) if (r <= cum[i]) return i;
    return cum.length - 1;
  }

  // ---- meshes ----------------------------------------------------------------
  _buildGeos() {
    this._geos = [_carGeo(), _truckGeo(), _busGeo(), _beamGeo(), _pedBodyGeo(), _pedHeadGeo()];
  }

  _buildMeshes() {
    const T = THREE;
    this._root = new T.Group();

    // Low metalness keeps the paint DIFFUSE-driven so the hemi+moon fill reads
    // in the dark (a 0.6 metal under night IBL goes near-black = "bodies vanish").
    // envMapIntensity 1.0 + a night emissive floor in update() keep silhouettes
    // legible when the sun is down.
    this._bodyMat = new T.MeshStandardMaterial({
      vertexColors: true, roughness: 0.38, metalness: 0.10, envMapIntensity: 1.0,
    });
    for (let k = 0; k < 3; k++) {
      const m = new T.InstancedMesh(this._geos[k], this._bodyMat, this._caps[k]);
      m.instanceMatrix.setUsage(T.DynamicDrawUsage);
      m.castShadow = true;
      m.frustumCulled = false;
      m.count = 0;
      this._bodyMesh.push(m);
      this._root.add(m);
    }

    // Pedestrians: capsule bodies take per-instance clothing colour; heads are a
    // fixed warm skin tone so every figure reads as "person" at street distance.
    this._pedMat = new T.MeshStandardMaterial({ roughness: 0.85, metalness: 0.0 });
    this._pedHeadMat = new T.MeshStandardMaterial({ color: 0xd8bfa3, roughness: 0.8, metalness: 0.0 });
    this._pedBodyMesh = new T.InstancedMesh(this._geos[4], this._pedMat, PED_TOTAL);
    this._pedBodyMesh.instanceMatrix.setUsage(T.DynamicDrawUsage);
    this._pedBodyMesh.castShadow = true;
    this._pedBodyMesh.frustumCulled = false;
    this._pedBodyMesh.count = 0;
    this._root.add(this._pedBodyMesh);
    this._pedHeadMesh = new T.InstancedMesh(this._geos[5], this._pedHeadMat, PED_TOTAL);
    this._pedHeadMesh.instanceMatrix.setUsage(T.DynamicDrawUsage);
    this._pedHeadMesh.castShadow = true;
    this._pedHeadMesh.frustumCulled = false;
    this._pedHeadMesh.count = 0;
    this._root.add(this._pedHeadMesh);

    this._lightGeo = new T.BoxGeometry(0.5, 0.2, 0.06);
    this._geos.push(this._lightGeo);
    // Unlit, tone-mapped OFF: values >1 pass straight into the bloom pass.
    this._lightMat = new T.MeshBasicMaterial({ toneMapped: false });
    const totalCap = this._caps[0] + this._caps[1] + this._caps[2];
    this._lightMesh = new T.InstancedMesh(this._lightGeo, this._lightMat, totalCap * 4);
    this._lightMesh.instanceMatrix.setUsage(T.DynamicDrawUsage);
    this._lightMesh.castShadow = false;
    this._lightMesh.frustumCulled = false;
    this._lightMesh.count = 0;
    this._root.add(this._lightMesh);

    // Headlight beam pools: additive, unlit, tone-mapped OFF, no depth write so
    // the flat quads never z-fight the asphalt. Ramped 0 by day in update().
    this._beamMat = new T.MeshBasicMaterial({
      vertexColors: true, transparent: true, blending: T.AdditiveBlending,
      depthWrite: false, toneMapped: false,
    });
    this._beamMesh = new T.InstancedMesh(this._geos[3], this._beamMat, totalCap);
    this._beamMesh.instanceMatrix.setUsage(T.DynamicDrawUsage);
    this._beamMesh.castShadow = false;
    this._beamMesh.frustumCulled = false;
    this._beamMesh.count = 0;
    this._beamMesh.renderOrder = 10;
    this._root.add(this._beamMesh);
  }

  // ---- fleet -----------------------------------------------------------------
  _buildFleet() {
    for (let k = 0; k < 3; k++) {
      for (let c = 0; c < this._fleetMix[k]; c++) this._makeVehicle(k, false);
    }
    for (const m of this._bodyMesh) if (m.instanceColor) m.instanceColor.needsUpdate = true;
    if (this._lightMesh.instanceColor) this._lightMesh.instanceColor.needsUpdate = true;
    this._beamMesh.count = this._vehicles.length;
    if (this._pedTotal > 0) this._buildPeds();
  }

  _buildPeds() {
    for (let i = 0; i < PED_TOTAL; i++) this._makePed();
    if (this._pedBodyMesh.instanceColor) this._pedBodyMesh.instanceColor.needsUpdate = true;
  }

  _makePed() {
    const eIdx = this._pickPedEdge();
    const p = {
      slot: this._peds.length,
      edge: eIdx,
      dir: this._rng.chance(0.5) ? 1 : -1,
      t: this._rng.next(),
      len: this._edges[eIdx].len,
      speed: this._rng.range(0.85, 1.55),   // m/s — stroll to brisk walk
      side: this._rng.chance(0.5) ? 1 : -1,
      lat: this._rng.range(PED_LAT[0], PED_LAT[1]),
      h: this._rng.range(0.92, 1.10),      // height variation
      phase: this._rng.range(0, 6.28),     // walk-bob phase
      state: 0,                            // 0 walk, 1 cross, 2 corner-turn, 3 linger
      fx: 0, fy: 0, fz: 0,                 // cross/turn start
      tx: 0, ty: 0, tz: 0,                 // cross/turn end
      gx: 0, gy: 0, gz: 0,                 // goal corner (turn end / linger spot)
      dist: 1,
      lingerT: 0,
      active: true,
    };
    const c = new THREE.Color().setStyle(this._pickPaint(PED_CLOTHES));
    this._pedBodyMesh.setColorAt(p.slot, c);
    this._pedBodyMesh.count = p.slot + 1;
    this._pedHeadMesh.count = p.slot + 1;
    this._peds.push(p);
    return p;
  }

  _makeVehicle(kind, atStart) {
    const eIdx = this._pickEdge();
    const e = this._edges[eIdx];
    const v = {
      kind,
      slot: this._count[kind]++,
      edge: eIdx,
      dir: this._rng.chance(0.5) ? 1 : -1,
      t: this._rng.next(),
      speed: kind === 0 ? this._rng.range(8, 14) : kind === 1 ? this._rng.range(6, 9) : this._rng.range(7, 10.5),
      len: e.len,
      laneOff: 1.8 + this._rng.range(0, 0.7), // right-hand lane + small natural spread
      lightBase: this._lightCursor,
      beamSlot: this._lightCursor >> 2,       // global vehicle index -> beam instance
      active: true,
    };
    this._lightCursor += 4;

    // per-instance paint
    const paints = kind === 0 ? CAR_PAINTS : kind === 1 ? TRUCK_PAINTS : BUS_PAINTS;
    const c = new THREE.Color().setStyle(this._pickPaint(paints));
    this._bodyMesh[kind].setColorAt(v.slot, c);
    this._bodyMesh[kind].count = v.slot + 1;

    // light quads: 0/1 = headlights (warm white), 2/3 = taillights (red)
    const head = new THREE.Color(1.05, 1.0, 0.92);
    const tail = new THREE.Color(1.0, 0.12, 0.07);
    const lb = v.lightBase;
    this._lightMesh.setColorAt(lb, head);
    this._lightMesh.setColorAt(lb + 1, head);
    this._lightMesh.setColorAt(lb + 2, tail);
    this._lightMesh.setColorAt(lb + 3, tail);
    this._lightMesh.count = lb + 4;

    if (atStart) v.t = this._rng.next() * 0.05;
    this._vehicles.push(v);
    return v;
  }

  _respawn(v) {
    const eIdx = this._pickEdge();
    v.edge = eIdx;
    v.dir = this._rng.chance(0.5) ? 1 : -1;
    v.t = this._rng.next() * 0.05;
    v.len = this._edges[eIdx].len;
  }

  _handoff(v, endArrival) {
    const list = this._trans[v.edge * 2 + endArrival];
    if (!list || !list.length) { this._respawn(v); return; }
    // mostly the best (straightest) option; occasionally the runner-up for variety
    const pick = (list.length > 1 && this._rng.chance(0.15)) ? list[1] : list[0];
    v.edge = pick.e;
    v.dir = pick.dir;
    v.len = this._edges[pick.e].len;
    v.t = 0;
  }

  // ---- per-frame --------------------------------------------------------------
  // light factor: 0 by day -> 1 at night; ramps over t 0.25-0.35 (dawn) and 0.85-0.95 (dusk)
  _lightFactor(t) {
    if (t < 0.25) return 1;
    if (t < 0.35) return 1 - smooth01((t - 0.25) * 10);
    if (t < 0.85) return 0;
    if (t < 0.95) return smooth01((t - 0.85) * 10);
    return 1;
  }

  update(dt, world) {
    if (!this._root || !this._ctx) return;
    const T = THREE;

    // keep our meshes in whichever scene is current (live <-> showcase)
    const target = this._showScene || this._ctx.scene;
    if (this._root.parent !== target) target.add(this._root);
    if (this._showScene) {
      if (this._terrainRoot && this._terrainRoot.parent !== this._showScene) this._showScene.add(this._terrainRoot);
      if (this._roadsRoot && this._roadsRoot.parent !== this._showScene) this._showScene.add(this._roadsRoot);
      this._stageLighting();
    }

    const t = this._ctx.clock.t;
    const lf = this._lightFactor(t);
    // brightness 0 (day) -> 5.0 (night); MeshBasicMaterial x toneMapped:false feeds the bloom pass
    this._lightMat.color.setScalar(7.0 * lf);
    // headlight beam pools: 0 by day, full at night
    this._beamMat.color.setScalar(1.9 * lf);

    // density by time of day: ~1.0 noon, ~0.55 dusk, ~0.35 deep night
    const e = Math.sin((t - 0.25) * Math.PI * 2);
    const activeBase = 0.35 + 0.65 * clamp(e + 0.25, 0, 1);
    const activeCount = Math.min(this._vehicles.length, Math.floor(this._vehicles.length * activeBase * this._userDensity));

    const terrain = this._terrainMod;
    for (let i = 0; i < this._vehicles.length; i++) {
      const v = this._vehicles[i];
      const act = i < activeCount;
      if (act !== v.active) {
        v.active = act;
        if (!act) this._writeHidden(v);
      }
      if (!act) continue;

      // advance travel progress (t: 0 at departure end -> 1 at arrival end);
      // dir only decides WHICH end is the arrival, so handoff + pts mapping are
      // uniform. A frame can at most cross one short stub (guard loops for safety).
      let tt = v.t + (v.speed * dt) / v.len;
      let guard = 0;
      while (guard++ < 3 && tt >= 1) {
        this._handoff(v, v.dir === 1 ? 1 : 0);
        tt = v.t;
      }
      v.t = tt;
      this._writeVehicle(v, terrain);
    }

    for (const m of this._bodyMesh) m.instanceMatrix.needsUpdate = true;
    this._lightMesh.instanceMatrix.needsUpdate = true;
    this._beamMesh.instanceMatrix.needsUpdate = true;
    if (world.traffic) world.traffic.active = activeCount;
  }

  // R = Ry(yaw) * Rx(pitch) * Rz(roll) columns written straight into the instance buffer.
  _writeVehicle(v, terrain) {
    const e = this._edges[v.edge];
    const pts = e.pts;
    const n = pts.length - 1;
    const f = (v.dir === 1 ? v.t : 1 - v.t) * n;
    let i = f | 0; if (i >= n) i = n - 1; if (i < 0) i = 0;
    const fr = f - i;
    const A = pts[i], B = pts[i + 1];
    if (!A || !B) { this._respawn(v); return; } // defensive: corrupt state -> respawn
    let x = A.x + (B.x - A.x) * fr;
    let y = A.y + (B.y - A.y) * fr;
    let z = A.z + (B.z - A.z) * fr;
    let fx = B.x - A.x, fz = B.z - A.z;
    const fh = Math.hypot(fx, fz) || 1;
    fx /= fh; fz /= fh;
    let dy = B.y - A.y;
    if (v.dir < 0) { fx = -fx; fz = -fz; dy = -dy; }

    // right-hand lane offset (right of travel = (fz, 0, -fx))
    const lane = v.laneOff * v.dir;
    x += fz * lane;
    z -= fx * lane;

    const yaw = Math.atan2(fx, fz);
    const pitch = Math.atan2(dy, fh);
    let roll = 0;
    if (terrain && typeof terrain.getHeight === 'function') {
      const rx = fz * 2.5, rz = -fx * 2.5;
      roll = Math.atan2(terrain.getHeight(x + rx, z + rz) - terrain.getHeight(x - rx, z - rz), 5);
    }
    const c = Math.cos(yaw), s = Math.sin(yaw);
    const cx = Math.cos(pitch), sx = Math.sin(pitch);
    const cz = Math.cos(roll), sz = Math.sin(roll);
    const r00 = c, r10 = 0, r20 = -s;
    const r01 = s * sx * cz + s * cx * sz, r11 = cx * cz - sx * sz, r21 = c * sx * cz + c * cx * sz;
    const r02 = -s * sx * sz + s * cx * cz, r12 = -cx * sz - sx * cz, r22 = -c * sx * sz + c * cx * cz;

    const arr = this._bodyMesh[v.kind].instanceMatrix.array;
    const o = v.slot * 16;
    arr[o] = r00; arr[o + 1] = r10; arr[o + 2] = r20; arr[o + 3] = 0;
    arr[o + 4] = r01; arr[o + 5] = r11; arr[o + 6] = r21; arr[o + 7] = 0;
    arr[o + 8] = r02; arr[o + 9] = r12; arr[o + 10] = r22; arr[o + 11] = 0;
    arr[o + 12] = x; arr[o + 13] = y; arr[o + 14] = z; arr[o + 15] = 1;

    // four light quads (head L/R, tail L/R) in the same rotation
    const L = LIGHT_OFF[v.kind];
    const la = this._lightMesh.instanceMatrix.array;
    const offs = [
      [-L.hx, L.hy,  L.hz], [L.hx, L.hy,  L.hz],
      [-L.tx, L.ty, -L.tz], [L.tx, L.ty, -L.tz],
    ];
    for (let k = 0; k < 4; k++) {
      const ox = offs[k][0], oy = offs[k][1], oz = offs[k][2];
      const lo = (v.lightBase + k) * 16;
      la[lo] = r00; la[lo + 1] = r10; la[lo + 2] = r20; la[lo + 3] = 0;
      la[lo + 4] = r01; la[lo + 5] = r11; la[lo + 6] = r21; la[lo + 7] = 0;
      la[lo + 8] = r02; la[lo + 9] = r12; la[lo + 10] = r22; la[lo + 11] = 0;
      la[lo + 12] = x + r00 * ox + r01 * oy + r02 * oz;
      la[lo + 13] = y + r10 * ox + r11 * oy + r12 * oz;
      la[lo + 14] = z + r20 * ox + r21 * oy + r22 * oz;
      la[lo + 15] = 1;
    }

    // headlight beam pool: same orientation + position as the body (local +Z forward)
    const ba = this._beamMesh.instanceMatrix.array;
    const bo = v.beamSlot * 16;
    ba[bo] = r00; ba[bo + 1] = r10; ba[bo + 2] = r20; ba[bo + 3] = 0;
    ba[bo + 4] = r01; ba[bo + 5] = r11; ba[bo + 6] = r21; ba[bo + 7] = 0;
    ba[bo + 8] = r02; ba[bo + 9] = r12; ba[bo + 10] = r22; ba[bo + 11] = 0;
    ba[bo + 12] = x; ba[bo + 13] = y; ba[bo + 14] = z; ba[bo + 15] = 1;
  }

  _writeHidden(v) {
    const arr = this._bodyMesh[v.kind].instanceMatrix.array;
    const o = v.slot * 16;
    arr.fill(0, o, o + 16);
    const la = this._lightMesh.instanceMatrix.array;
    const lo = v.lightBase * 16;
    la.fill(0, lo, lo + 64);
    const ba = this._beamMesh.instanceMatrix.array;
    const bo = v.beamSlot * 16;
    ba.fill(0, bo, bo + 16);
  }

  // ---- pedestrians -------------------------------------------------------------
  // Day curve: peak at noon, a second (slightly lower) evening peak ~19:00,
  // sparse at deep night. t: 0=midnight, 0.5=noon, 0.75=18:00.
  _pedDayFactor(t) {
    const d = (a, b) => { const d2 = Math.abs(a - b); return Math.min(d2, 1 - d2); };
    const noon = Math.exp(-(d(t, 0.50) ** 2) / 0.018);
    const eve = 0.85 * Math.exp(-(d(t, 0.79) ** 2) / 0.006);
    return 0.12 + 0.88 * Math.max(noon, eve);
  }

  _stepPed(p, dt) {
    if (p.state === 3) { // linger on a corner
      p.lingerT -= dt;
      if (p.lingerT <= 0) p.state = 0;
      return;
    }
    if (p.state === 1 || p.state === 2) { // crossing the street / cutting a corner
      p.phase += p.speed * dt * 4.2;
      p.t += (p.speed * dt) / p.dist;
      if (p.t >= 1) {
        p.t = 0;
        const wantLinger = p.lingerT > 0;
        // the cross ends on the far sidewalk of the INCOMING street; if the
        // departure corner (outgoing street) is further away, cut to it first
        const d2 = Math.hypot(p.gx - p.tx, p.gz - p.tz);
        if (d2 > 2.5) {
          p.fx = p.tx; p.fy = p.ty; p.fz = p.tz;
          p.tx = p.gx; p.ty = p.gy; p.tz = p.gz;
          p.dist = d2;
          p.state = 2;
        } else if (wantLinger) {
          p.lingerT = p._lingDur;
          p.state = 3;
        } else {
          p.state = 0;
        }
      }
      return;
    }
    // walk
    p.phase += p.speed * dt * 4.2;
    p.t += (p.speed * dt) / p.len;
    let guard = 0;
    while (guard++ < 3 && p.t >= 1) {
      this._pedArrive(p);
      p.t = 0;
      if (p.state !== 0) break;
    }
  }

  // Arrived at a node: pick the next edge (straight preferred, like vehicles),
  // maybe cross the street at the crosswalk (flip sidewalk side), and if the
  // departure corner sits far from the arrival point, queue a corner cut.
  _pedArrive(p) {
    const endArr = p.dir === 1 ? 1 : 0;
    const e = this._edges[p.edge];
    const list = this._trans[e.nodeA * 0 + p.edge * 2 + endArr];
    let eIdx = p.edge, dir = p.dir === 1 ? -1 : 1; // dead end -> U-turn
    if (list && list.length) {
      const pick = (list.length > 1 && this._rng.chance(0.2)) ? list[1] : list[0];
      eIdx = pick.e; dir = pick.dir;
    }
    const ne = this._edges[eIdx];

    const node = endArr === 1 ? e.pts[e.pts.length - 1] : e.pts[0];
    const inU = endArr === 1 ? e.bUnit : { x: -e.aUnit.x, z: -e.aUnit.z };
    const nrx = inU.z, nrz = -inU.x; // right-of-travel normal on the incoming street
    // arrival corner (where the pedestrian currently stands)
    const cx = node.x + nrx * (p.side * p.lat);
    const cz = node.z + nrz * (p.side * p.lat);
    const cy = node.y - 0.05;

    const doCross = this._rng.chance(0.30);
    const sNew = doCross ? -p.side : p.side;

    // departure corner on the outgoing street
    const na = ne.pts[0];
    const outU = dir === 1 ? ne.aUnit : { x: -ne.bUnit.x, z: -ne.bUnit.z };
    const onx = outU.z, onz = -outU.x;
    const gx = na.x + onx * (sNew * p.lat);
    const gz = na.z + onz * (sNew * p.lat);
    const gy = na.y - 0.05;

    // commit the routing handoff
    p.edge = eIdx; p.dir = dir; p.len = ne.len; p.t = 0; p.side = sNew;
    p.gx = gx; p.gy = gy; p.gz = gz;

    if (doCross) {
      // cross the incoming street at the node (the crosswalk sits right here)
      p.fx = cx; p.fy = cy; p.fz = cz;
      p.tx = node.x + nrx * (-p.side * p.lat);
      p.ty = cy;
      p.tz = node.z + nrz * (-p.side * p.lat);
      p.dist = Math.hypot(p.tx - p.fx, p.tz - p.fz) || 1;
      p.state = 1;
    } else if (Math.hypot(gx - cx, gz - cz) > 2.5) {
      p.fx = cx; p.fy = cy; p.fz = cz;
      p.tx = gx; p.ty = gy; p.tz = gz;
      p.dist = Math.hypot(gx - cx, gz - cz);
      p.state = 2;
    }
    // linger on corners near commercial nodes (queued; consumed when in place)
    const nc = this._nodeComm[endArr === 1 ? e.nodeB : e.nodeA];
    if (nc > 0 && this._rng.chance(0.30 * nc)) {
      p._lingDur = this._rng.range(2.5, 9);
      if (p.state === 0) { p.state = 3; p.lingerT = p._lingDur; }
      else p.lingerT = p._lingDur; // armed for the cross/turn completion
    }
  }

  _writePed(p) {
    const e = this._edges[p.edge];
    let x, y, z;
    if (p.state === 1 || p.state === 2) {
      x = p.fx + (p.tx - p.fx) * p.t;
      y = p.fy + (p.ty - p.fy) * p.t;
      z = p.fz + (p.tz - p.fz) * p.t;
    } else if (p.state === 3) {
      x = p.gx; y = p.gy; z = p.gz;
    } else {
      const pts = e.pts;
      const n = pts.length - 1;
      const f = (p.dir === 1 ? p.t : 1 - p.t) * n;
      let i = f | 0; if (i >= n) i = n - 1; if (i < 0) i = 0;
      const fr = f - i;
      const A = pts[i], B = pts[i + 1];
      x = A.x + (B.x - A.x) * fr;
      y = A.y + (B.y - A.y) * fr;
      z = A.z + (B.z - A.z) * fr;
      let fx = B.x - A.x, fz = B.z - A.z;
      const fh = Math.hypot(fx, fz) || 1;
      fx /= fh; fz /= fh;
      if (p.dir < 0) { fx = -fx; fz = -fz; }
      const off = p.side * p.lat;
      x += fz * off;   // right-of-travel offset (right = (fz, -fx))
      z -= fx * off;
      y -= 0.05;       // onto the sidewalk slab (0.05 below the asphalt line)
    }
    // subtle walk bob (2.5 cm) — life at close range, invisible from above
    const moving = p.state !== 3;
    const bob = moving ? 0.025 * Math.abs(Math.sin(p.phase)) : 0;
    y += bob;

    const h = p.h;
    const ab = this._pedBodyMesh.instanceMatrix.array;
    const o = p.slot * 16;
    ab[o] = h; ab[o + 1] = 0; ab[o + 2] = 0; ab[o + 3] = 0;
    ab[o + 4] = 0; ab[o + 5] = h; ab[o + 6] = 0; ab[o + 7] = 0;
    ab[o + 8] = 0; ab[o + 9] = 0; ab[o + 10] = h; ab[o + 11] = 0;
    ab[o + 12] = x; ab[o + 13] = y; ab[o + 14] = z; ab[o + 15] = 1;
    const ah = this._pedHeadMesh.instanceMatrix.array;
    ah[o] = h; ah[o + 1] = 0; ah[o + 2] = 0; ah[o + 3] = 0;
    ah[o + 4] = 0; ah[o + 5] = h; ah[o + 6] = 0; ah[o + 7] = 0;
    ah[o + 8] = 0; ah[o + 9] = 0; ah[o + 10] = h; ah[o + 11] = 0;
    ah[o + 12] = x; ah[o + 13] = y; ah[o + 14] = z; ah[o + 15] = 1;
  }

  _writePedHidden(p) {
    const o = p.slot * 16;
    this._pedBodyMesh.instanceMatrix.array.fill(0, o, o + 16);
    this._pedHeadMesh.instanceMatrix.array.fill(0, o, o + 16);
  }

  // ---- public API --------------------------------------------------------------
  setDensity(d) {
    this._userDensity = clamp(Number(d) || 0, 0, 1);
    if (this._ctx) this._ctx.events.emit('traffic:density', { d: this._userDensity });
  }

  spawn(kind) {
    if (!this._root || !this._edges.length) return null;
    const k = kind === 'truck' ? 1 : kind === 'bus' ? 2 : 0;
    if (this._count[k] >= this._caps[k]) return null;
    const v = this._makeVehicle(k, true);
    return v;
  }

  stats() {
    return {
      drawCalls: 5,
      vehicles: this._vehicles.length,
      active: this._vehicles.length ? this._vehicles.filter((v) => v.active).length : 0,
      notes: '3 instanced bodies (car/truck/bus, real massing + vertex-coloured glass/wheels) + 1 instanced light-quad mesh (head/tail) + 1 additive instanced headlight-beam mesh (night-ramped); graph rebuilt with crossing splits; O(n) update',
    };
  }

  // ---- showcase ----------------------------------------------------------------
  showcase(scene, world, ctx) {
    this._showScene = scene;
    // remove foreign staging meshes (the flat grass ground) — terrain provides the ground
    for (const c of [...scene.children]) {
      if (c.isMesh) scene.remove(c);
    }
    // MOVE the terrain + roads roots in (never clone)
    const reg = ctx.registry;
    const terrain = reg && reg.get('terrain');
    if (terrain && terrain._root) {
      terrain._root.removeFromParent();
      scene.add(terrain._root);
      this._terrainRoot = terrain._root;
    }
    const roads = reg && reg.get('roads');
    if (roads && roads._root) {
      roads._root.removeFromParent();
      scene.add(roads._root);
      this._roadsRoot = roads._root;
    }
    scene.add(this._root);

    // Push the aerial fog back so the ~240 m city (and its sub-metre cars) stays
    // crisp at the 600 m+ aerial preset; the default 300 m start fogs the fleet
    // to ~30 % and makes it vanish from above.
    if (scene.fog && scene.fog.isFog) { scene.fog.near = 520; scene.fog.far = 2800; }

    // borrow IBL
    const env = reg && reg.get('environment');
    if (env && env._ibRT) { scene.environment = env._ibRT.texture; scene.environmentIntensity = 0.55; }

    // our own sun
    for (const o of this._showExtras) o.removeFromParent();
    this._showExtras.length = 0;
    this._showSun = null;
    this._showHemi = null;
    this._showMoon = null;
    for (const c of scene.children) if (c.isHemisphereLight) this._showHemi = c;
    const sun = new ctx.three.DirectionalLight(0xfff6ec, 2.0);
    sun.position.set(200, 300, 140);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = -260; sc.right = 260; sc.top = 260; sc.bottom = -260; sc.near = 20; sc.far = 900;
    sc.updateProjectionMatrix();
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.9;
    scene.add(sun, sun.target);
    this._showExtras.push(sun, sun.target);
    this._showSun = sun;

    // cool moon fill so car BODIES keep form + read in the dark (never shadow-casting)
    const moon = new ctx.three.DirectionalLight(0x8fa4d6, 0.0);
    moon.position.set(-260, 220, -180);
    scene.add(moon, moon.target);
    this._showExtras.push(moon, moon.target);
    this._showMoon = moon;

    this._bgDay = this._bgDay || new ctx.three.Color(0x9db6d4);
    this._bgNight = this._bgNight || new ctx.three.Color(0x070d18);
    this._stageLighting();
  }

  // follow the clock in the showcase: sun -> 0, moon + a solid hemi floor stay up
  // at night so car bodies read in the dark; sky/fog -> near-black at night
  _stageLighting() {
    const s = this._showScene;
    if (!s) return;
    const lf = this._lightFactor(this._ctx.clock.t);
    const day = 1 - lf;
    if (this._showSun) this._showSun.intensity = 2.0 * day;
    if (this._showMoon) this._showMoon.intensity = 0.85 * lf;
    // night hemi floor kept well above black so paint never goes pure black
    if (this._showHemi) this._showHemi.intensity = 0.42 * (0.40 + 0.60 * day);
    if (s.background && s.background.isColor) s.background.copy(this._bgDay).lerp(this._bgNight, lf);
    if (s.fog && s.fog.isFog) s.fog.color.copy(this._bgDay).lerp(this._bgNight, lf);
  }

  dispose() {
    const live = this._ctx && this._ctx.scene;
    if (live) {
      if (this._terrainRoot) { live.add(this._terrainRoot); this._terrainRoot = null; } // give terrain back
      if (this._roadsRoot) { live.add(this._roadsRoot); this._roadsRoot = null; }       // give roads back
    }
    for (const o of this._showExtras) o.removeFromParent();
    this._showExtras.length = 0;
    this._showSun = null;
    this._showHemi = null;
    if (this._root) this._root.removeFromParent();
    for (const g of this._geos) if (g) g.dispose();
    if (this._bodyMat) this._bodyMat.dispose();
    if (this._lightMat) this._lightMat.dispose();
    if (this._beamMat) this._beamMat.dispose();
    this._root = null;
    this._showScene = null;
  }
}
