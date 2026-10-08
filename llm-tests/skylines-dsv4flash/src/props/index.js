// ============================================================================
// props module — Skylines
// Procedural CC0-style street furniture + vegetation, fully seeded.
//
// Owns ONLY src/props/. All randomness derives from a local mulberry32 RNG keyed
// off the world seed (isolated from other modules' draw order -> deterministic).
//
// Instanced rendering: one InstancedMesh per prop part, shared materials and a
// shared canvas foliage atlas. ~25 draw calls for the whole module.
//
// Night behaviour: streetlights + lamp posts switch on emissive heads and drop a
// warm additive light pool on the ground, driven by time-of-day in update().
// ============================================================================
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export const id = 'props';

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;

// local seeded RNG isolated from the shared world.rng draw order
function mulberry(seed0) {
  let a = seed0 >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function localRNG(seed) { return mulberry((seed ^ 0x51ed270b) >>> 0); }

// deterministic value-noise for procedural textures (independent of rng draws)
function hnoise(x, y, s) {
  let n = ((x * 374761393 + y * 668265263 + (s | 0) * 2246822519) >>> 0);
  n = Math.imul(n ^ (n >>> 13), 1274126177);
  return (((n ^ (n >>> 16)) >>> 0) / 4294967296);
}

// ---------------------------------------------------------------------------
// Tree species table
//   broad: blobs of canopy spheres near the crown.
//   conifer: stacked cone tiers.
// ---------------------------------------------------------------------------
const SPECIES = [
  {
    name: 'oak', kind: 'broad',
    trunkH: 3.0, t0: 0.30, t1: 0.16, branches: 3,
    blobs: [[0, 3.7, 0], [1.2, 3.0, 0.3], [-1.0, 2.9, -0.8], [0.4, 4.3, 0.6]],
    blobR: 1.25, base: 0x5c7d34,
  },
  {
    name: 'maple', kind: 'broad',
    trunkH: 4.2, t0: 0.26, t1: 0.11, branches: 3,
    blobs: [[0, 5.0, 0], [1.05, 4.4, -0.35], [-0.85, 4.3, 0.9], [0, 5.8, 0.15]],
    blobR: 1.10, base: 0x6d8f3a,
  },
  {
    name: 'pine', kind: 'conifer',
    trunkH: 4.6, t0: 0.20, t1: 0.07, branches: 2,
    tiers: [[2.2, 2.7, 2.5], [3.5, 2.4, 1.9], [4.7, 2.3, 1.2]],
    base: 0x2f6b33,
  },
  {
    name: 'birch', kind: 'broad',
    trunkH: 5.2, t0: 0.15, t1: 0.07, branches: 2,
    blobs: [[0, 6.4, 0], [-1.0, 6.1, -0.4], [0.9, 6.0, 0.35]],
    blobR: 0.95, base: 0x7fae53,
  },
];

// ---------------------------------------------------------------------------
// Procedural canvas textures (deterministic, shared across instanced parts)
// ---------------------------------------------------------------------------
function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return { c, ctx: c.getContext('2d') };
}

function buildLeafTexture(seed) {
  // mottled leafy green with lighter speckle -> reads as foliage on canopy blobs
  const N = 128;
  const { c, ctx } = makeCanvas(N, N);
  const img = ctx.createImageData(N, N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let n = hnoise(x / N * 6, y / N * 6, seed + 11) * 0.7;
      n += hnoise(x / N * 18, y / N * 18, seed + 29) * 0.3;
      const g = 92 + n * 70;             // green spread
      const r = g * (0.62 + hnoise(x, y, seed + 5) * 0.28);
      const b = g * (0.48 + hnoise(x * 2, y * 2, seed + 7) * 0.25);
      const i = (y * N + x) * 4;
      img.data[i] = clamp(r, 0, 255);
      img.data[i + 1] = clamp(g, 0, 255);
      img.data[i + 2] = clamp(b, 0, 255);
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(2, 2);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

function buildBarkTexture(seed) {
  // vertical streaky bark
  const W = 128, H = 64;
  const { c, ctx } = makeCanvas(W, H);
  const img = ctx.createImageData(W, H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const streak = hnoise(Math.floor(x / 6), y, seed + 3) * 0.5
                   + hnoise(x, y / 6, seed + 8) * 0.5;
      const g = 118 + streak * 46;
      const r = g * (1.05 + (hnoise(x, y, seed + 2) - 0.5) * 0.3);
      const b = g * 0.55;
      const i = (y * W + x) * 4;
      img.data[i] = clamp(r, 0, 255);
      img.data[i + 1] = clamp(g, 0, 255);
      img.data[i + 2] = clamp(b, 0, 255);
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function buildGlowTexture(seed) {
  // warm radial gradient for the night light pool (alpha -> edge)
  const N = 128;
  const { c, ctx } = makeCanvas(N, N);
  const g = ctx.createRadialGradient(N / 2, N / 2, 0, N / 2, N / 2, N / 2);
  g.addColorStop(0.0, 'rgba(255,214,150,1)');
  g.addColorStop(0.35, 'rgba(255,178,102,0.85)');
  g.addColorStop(0.75, 'rgba(240,140,60,0.28)');
  g.addColorStop(1.0, 'rgba(200,110,40,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, N, N);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
const _m = new THREE.Matrix4();
function placed(geo, tx, ty, tz, rotX = 0, rotY = 0, rotZ = 0, sx = 1, sy = 1, sz = 1) {
  const g = geo.clone();
  _m.makeScale(sx, sy, sz);
  g.applyMatrix4(_m);
  _m.makeRotationX(rotX); g.applyMatrix4(_m);
  _m.makeRotationY(rotY); g.applyMatrix4(_m);
  _m.makeRotationZ(rotZ); g.applyMatrix4(_m);
  _m.makeTranslation(tx, ty, tz); g.applyMatrix4(_m);
  return g;
}

function buildTrunkGeo(sp) {
  const parts = [];
  parts.push(placed(new THREE.CylinderGeometry(sp.t1, sp.t0, sp.trunkH, 7),
    0, sp.trunkH / 2, 0));
  // branch stubs near the crown
  for (let i = 0; i < sp.branches; i++) {
    const a = (i / sp.branches) * Math.PI * 2;
    const by = sp.trunkH * (0.68 + (i % 2) * 0.15);
    parts.push(placed(new THREE.CylinderGeometry(sp.t1 * 0.5, sp.t1 * 0.8, 0.9, 5),
      Math.cos(a) * sp.t0 * 2.4, by, Math.sin(a) * sp.t0 * 2.4,
      0, a + Math.PI / 2, 0, 1, 1, 1));
  }
  const merged = mergeGeometries(parts);
  return merged || new THREE.BufferGeometry();
}

function buildCanopyGeo(sp) {
  const parts = [];
  if (sp.kind === 'conifer') {
    for (const [y, hgt, r] of sp.tiers) {
      parts.push(placed(new THREE.ConeGeometry(r, hgt, 8), 0, y - hgt / 2, 0));
    }
  } else {
    const blobs = sp.blobs || [[0, sp.trunkH + 1.4, 0]];
    for (const [x, y, z] of blobs) {
      parts.push(placed(new THREE.SphereGeometry(sp.blobR || 1.1, 7, 5),
        x, y, z, 0, 0, 0, 1, 0.78, 1)); // slight vertical squash
    }
  }
  const merged = mergeGeometries(parts);
  return merged || new THREE.BufferGeometry();
}

function buildBenchGeo() {
  const parts = [];
  parts.push(placed(new THREE.BoxGeometry(1.6, 0.08, 0.5), 0, 0.46, 0));           // seat
  parts.push(placed(new THREE.BoxGeometry(1.6, 0.45, 0.06), 0, 0.9, -0.24));       // backrest
  parts.push(placed(new THREE.BoxGeometry(0.09, 0.42, 0.42), -0.72, 0.21, 0.05));  // legs
  parts.push(placed(new THREE.BoxGeometry(0.09, 0.42, 0.42), 0.72, 0.21, 0.05));
  return mergeGeometries(parts);
}

function buildHydrantGeo() {
  const parts = [];
  parts.push(placed(new THREE.CylinderGeometry(0.15, 0.17, 0.55, 8), 0, 0.27, 0));
  parts.push(placed(new THREE.CylinderGeometry(0.09, 0.09, 0.12, 8), 0, 0.61, 0));
  parts.push(placed(new THREE.BoxGeometry(0.06, 0.16, 0.18), -0.14, 0.5, 0));      // nozzle
  return mergeGeometries(parts);
}

function buildFenceGeo() {
  const parts = [];
  parts.push(placed(new THREE.BoxGeometry(0.07, 1.05, 0.07), -0.97, 0.52, 0));
  parts.push(placed(new THREE.BoxGeometry(0.07, 1.05, 0.07), 0.97, 0.52, 0));
  parts.push(placed(new THREE.BoxGeometry(2.0, 0.05, 0.03), 0, 0.92, 0));          // top rail
  parts.push(placed(new THREE.BoxGeometry(2.0, 0.05, 0.03), 0, 0.5, 0));           // mid rail
  return mergeGeometries(parts);
}

function buildStreetPoleGeo() {
  const parts = [];
  parts.push(placed(new THREE.CylinderGeometry(0.07, 0.10, 7.3, 6), 0, 3.65, 0));
  parts.push(placed(new THREE.CylinderGeometry(0.05, 0.05, 1.5, 5),
    0.75, 7.25, 0, 0, 0, Math.PI / 2));                                            // arm
  return mergeGeometries(parts);
}

function buildStreetHeadGeo() {
  const parts = [];
  parts.push(placed(new THREE.BoxGeometry(0.16, 0.10, 0.5), 1.55, 7.28, 0));
  return mergeGeometries(parts);
}

function buildLampBodyGeo() {
  const parts = [];
  parts.push(placed(new THREE.CylinderGeometry(0.05, 0.07, 3.0, 6), 0, 1.5, 0));
  parts.push(placed(new THREE.BoxGeometry(0.05, 0.06, 0.14), 0.02, 2.98, 0));      // small arm
  return mergeGeometries(parts);
}

function buildLampHeadGeo() {
  const parts = [];
  parts.push(placed(new THREE.BoxGeometry(0.10, 0.08, 0.18), 0.06, 3.04, 0));
  return mergeGeometries(parts);
}

function buildBusFrameGeo() {
  const parts = [];
  parts.push(placed(new THREE.BoxGeometry(0.07, 2.3, 0.07), -1.1, 1.15, 0));
  parts.push(placed(new THREE.BoxGeometry(0.07, 2.3, 0.07), 1.1, 1.15, 0));
  parts.push(placed(new THREE.BoxGeometry(2.34, 0.09, 1.1), 0, 2.32, -0.45));      // roof
  parts.push(placed(new THREE.BoxGeometry(0.07, 0.5, 1.0), 0, 1.05, -0.9));        // bench back
  parts.push(placed(new THREE.BoxGeometry(1.4, 0.06, 0.45), 0, 0.45, -0.55));      // seat
  return mergeGeometries(parts);
}

function buildBusGlassGeo() {
  // clear rear/roof panels -> transparent instanced mesh
  const parts = [];
  parts.push(placed(new THREE.BoxGeometry(2.1, 1.6, 0.04), 0, 1.4, -0.62));        // back pane
  parts.push(placed(new THREE.BoxGeometry(0.9, 0.06, 1.05), 1.5, 2.28, -0.45));    // roof light strip
  return mergeGeometries(parts);
}

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------
const state = {
  built: false,
  group: null,
  rng: null,
  headMats: [],     // { mat, glow }
  streetHead: null,
  lampHead: null,
  glowBig: null,
  glowSmall: null,
  counts: {},
};

function heightAt(world, x, z) {
  try {
    const f = world && world.terrain && world.terrain.heightAt;
    if (typeof f === 'function') { const v = f(x, z); return isFinite(v) ? v : 0; }
  } catch (e) { /* absent */ }
  return 0;
}

// point-to-segment distance
function distToSeg(px, pz, s) {
  const ax = s.from[0], az = s.from[1], bx = s.to[0], bz = s.to[1];
  const vx = bx - ax, vz = bz - az;
  const l2 = vx * vx + vz * vz || 1;
  let t = ((px - ax) * vx + (pz - az) * vz) / l2;
  t = clamp(t, 0, 1);
  return Math.hypot(px - (ax + t * vx), pz - (az + t * vz));
}

// segment intersection -> {x,z} or null
function segIntersect(sa, sb) {
  const ax = sa.from[0], az = sa.from[1], bx = sa.to[0], bz = sa.to[1];
  const cx = sb.from[0], cz = sb.from[1], dx = sb.to[0], dz = sb.to[1];
  const denom = (bx - ax) * (dz - cz) - (bz - az) * (dx - cx);
  if (Math.abs(denom) < 1e-9) return null;
  const t = ((cx - ax) * (dz - cz) - (cz - az) * (dx - cx)) / denom;
  const u = ((cx - ax) * (bz - az) - (cz - az) * (bx - ax)) / denom;
  if (t < 1e-4 || t > 1 - 1e-4 || u < 1e-4 || u > 1 - 1e-4) return null;
  return { x: ax + t * (bx - ax), z: az + t * (bz - az) };
}

function perp(ax, az, bx, bz) {
  const dx = bx - ax, dz = bz - az;
  const l = Math.hypot(dx, dz) || 1;
  return { x: dz / l, z: -dx / l };
}
function dirOf(s) {
  const dx = s.to[0] - s.from[0], dz = s.to[1] - s.from[1];
  const l = Math.hypot(dx, dz) || 1;
  return { x: dx / l, z: dz / l };
}

// ---------------------------------------------------------------------------
// Placement
// ---------------------------------------------------------------------------
function collectCrossings(roads) {
  const pts = [];
  for (let i = 0; i < roads.length; i++) {
    for (let j = i + 1; j < roads.length; j++) {
      const c = segIntersect(roads[i], roads[j]);
      if (c) pts.push({ x: c.x, z: c.z });
    }
  }
  // dedupe close points
  const out = [];
  for (const p of pts) {
    let dup = false;
    for (const q of out) if (Math.hypot(p.x - q.x, p.z - q.z) < 2.0) { dup = true; break; }
    if (!dup) out.push(p);
  }
  return out;
}

function nearAnyCrossing(x, z, crossings, r) {
  for (const c of crossings) if (Math.hypot(x - c.x, z - c.z) < r) return true;
  return false;
}

function placeEverything(world) {
  const scene = world.scene;
  const rng = state.rng;
  const roads = Array.isArray(world.roads) ? world.roads : [];
  const buildings = Array.isArray(world.buildings) ? world.buildings : [];

  function occupied(x, z, rad) {
    if (nearAnyCrossing(x, z, crossings, Math.max(14, rad))) return true;
    for (const b of buildings) {
      if (typeof b.x === 'number' && typeof b.z === 'number') {
        if (Math.hypot(x - b.x, z - b.z) < rad + 6) return true;
      }
    }
    return false;
  }

  const crossings = collectCrossings(roads);

  // ---- roadside trees ----------------------------------------------------
  const treeBuckets = SPECIES.map(() => []);
  for (const s of roads) {
    const len = Math.hypot(s.to[0] - s.from[0], s.to[1] - s.from[1]);
    if (!(len > 4)) continue;
    const d = dirOf(s), p = perp(s.from[0], s.from[1], s.to[0], s.to[1]);
    const n = Math.floor(len / 16);
    let side = rng() < 0.5 ? -1 : 1;
    for (let i = 0; i < n; i++) {
      if (i % 2 === 0) side = -side;
      const t = (i + 0.5) / n + (rng() - 0.5) * 0.25;
      if (t <= 0 || t >= 1) continue;
      if (rng() < 0.16) continue;                       // gaps for variety
      const px = s.from[0] + d.x * len * t, pz = s.from[1] + d.z * len * t;
      if (nearAnyCrossing(px, pz, crossings, 17)) continue;
      const off = (s.width || 14) / 2 + 2.0 + rng() * 1.3;
      const x = px + p.x * side * off, z = pz + p.z * side * off;
      if (occupied(x, z, 4)) continue;
      // pick species by weighted roll
      let sp = Math.floor(rng() * SPECIES.length);
      if (rng() < 0.12) sp = 3;                          // a few birch accents
      treeBuckets[sp % SPECIES.length].push({ x, z });
    }
  }

  // ---- block/park interior trees -----------------------------------------
  const parkSpots = [];
  let interiors = 0;
  for (let k = 0; k < 900 && interiors < 230; k++) {
    const x = -460 + rng() * 920, z = -460 + rng() * 920;
    // must be well inside a block (away from any road centreline)
    let minD = Infinity;
    for (const s of roads) { const dd = distToSeg(x, z, s); if (dd < minD) minD = dd; }
    if (minD < 30) continue;
    if (nearAnyCrossing(x, z, crossings, 22)) continue;
    let sp = Math.floor(rng() * SPECIES.length);
    treeBuckets[sp].push({ x, z });
    interiors++;
    if (rng() < 0.20) parkSpots.push({ x, z });          // small green cluster
  }

  // ---- streetlights along roads -------------------------------------------
  const streetLights = [];
  for (const s of roads) {
    const len = Math.hypot(s.to[0] - s.from[0], s.to[1] - s.from[1]);
    if (!(len > 10)) continue;
    const d = dirOf(s), p = perp(s.from[0], s.from[1], s.to[0], s.to[1]);
    const n = Math.floor(len / 26);
    let side = rng() < 0.5 ? -1 : 1;
    for (let i = 0; i < n; i++) {
      if (i % 2 === 0) side = -side;
      const t = (i + 0.5) / n + (rng() - 0.5) * 0.3;
      if (t <= 0 || t >= 1) continue;
      if (nearAnyCrossing(s.from[0] + d.x * len * t, s.from[1] + d.z * len * t, crossings, 16)) continue;
      const off = (s.width || 14) / 2 + 1.5;
      const x = s.from[0] + d.x * len * t + p.x * side * off;
      const z = s.from[1] + d.z * len * t + p.z * side * off;
      // orient arm toward road centreline (perp inward)
      const ix = -p.x * side, iz = -p.z * side;
      const rotY = Math.atan2(-iz, ix);
      streetLights.push({ x, z, rotY });
    }
  }

  // ---- furniture near intersections ----------------------------------------
  const benchesT = [], hydrantsT = [], busT = [], lampT = [];
  for (const c of crossings) {
    const a = Math.atan2(c.z, c.x);                     // deterministic-ish corner
    for (let k = 0; k < 4; k++) {
      const ang = a + k * Math.PI / 2;
      const cx = c.x + Math.cos(ang) * 8, cz = c.z + Math.sin(ang) * 8;
      if (occupied(cx, cz, 5)) continue;
      const roll = rng();
      if (roll < 0.45) hydrantsT.push({ x: cx, z: cz, rotY: ang });
      else if (roll < 0.62) benchesT.push({ x: cx, z: cz, rotY: ang + Math.PI / 2 });
      else if (roll < 0.74) lampT.push({ x: cx, z: cz });
      else if (roll < 0.80) busT.push({ x: c.x, z: c.z, rotY: a });
    }
  }

  // ---- furniture in parks ---------------------------------------------------
  for (const p of parkSpots) {
    if (rng() < 0.8) benchesT.push({ x: p.x + rng() * 2 - 1, z: p.z + rng() * 2 - 1, rotY: rng() * Math.PI });
    if (rng() < 0.6) lampT.push({ x: p.x, z: p.z });
  }

  // ---- build all instanced meshes ------------------------------------------
  state.counts = {
    trees: interiors + treeBuckets.reduce((a, b) => a + b.length, 0),
    streetLights: streetLights.length,
    benches: benchesT.length,
    hydrants: hydrantsT.length,
    busStops: busT.length,
    lampPosts: lampT.length,
    parks: parkSpots.length,
    crossings: crossings.length,
  };

  buildTrees(scene, treeBuckets);
  buildStreetlights(scene, streetLights);
  buildLampPosts(scene, lampT);
  buildBenches(scene, benchesT);
  buildHydrants(scene, hydrantsT);
  buildBusStops(scene, busT);
  buildFences(scene, parkSpots);
}

// generic instanced builder
function inst(group, geo, mat, transforms) {
  const n = transforms.length;
  if (n === 0) return null;
  const mesh = new THREE.InstancedMesh(geo, mat, n);
  mesh.castShadow = true;
  mesh.frustumCulled = false;   // instances spread widely; one call each regardless
  for (let i = 0; i < n; i++) {
    const T = transforms[i];
    mesh.setMatrixAt(i, T.matrix);
    if (T.color) mesh.setColorAt(i, T.color);
  }
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  group.add(mesh);
  return mesh;
}

function txAt(x, y, z, rotY, scale) {
  const q = new THREE.Quaternion().setFromAxisAngle(_u.set(0, 1, 0), rotY || 0);
  return new THREE.Matrix4().compose(_p2.set(x, y, z), q, _s2.setScalar(scale));
}
const _u = new THREE.Vector3();
const _p2 = new THREE.Vector3();
const _s2 = new THREE.Vector3();

function buildTrees(scene, buckets) {
  const leafTex = state.leafTex;
  const barkTex = state.barkTex;
  for (let si = 0; si < SPECIES.length; si++) {
    const sp = SPECIES[si];
    const spots = buckets[si];
    if (!spots.length) continue;

    const trunkGeo = buildTrunkGeo(sp);
    const canopyGeo = buildCanopyGeo(sp);

    const barkMat = new THREE.MeshStandardMaterial({
      map: barkTex, color: 0xc8b39a, roughness: 0.92, metalness: 0.0,
    });
    const leafMat = new THREE.MeshStandardMaterial({
      map: leafTex, color: sp.base, roughness: 0.95, metalness: 0.0,
    });

    const transforms = [];
    for (const s of spots) {
      const y = heightAt(worldRef, s.x, s.z);
      const scale = 0.85 + state.rng() * 0.4;
      const rotY = state.rng() * Math.PI * 2;
      transforms.push({ matrix: txAt(s.x, y, s.z, rotY, scale) });
    }

    // trunks
    const tmesh = inst(scene, trunkGeo, barkMat, transforms);
    if (tmesh) {
      for (let i = 0; i < transforms.length; i++) {
        const v = 0.82 + state.rng() * 0.25;
        tmesh.setColorAt(i, new THREE.Color(v * 0.9, v * 0.78, v * 0.58));
      }
      tmesh.instanceColor.needsUpdate = true;
    }
    // canopies
    const cmesh = inst(scene, canopyGeo, leafMat, transforms);
    if (cmesh) {
      for (let i = 0; i < transforms.length; i++) {
        const g = 0.85 + state.rng() * 0.35;
        cmesh.setColorAt(i, new THREE.Color(g * 0.55, g, g * 0.5));
      }
      cmesh.instanceColor.needsUpdate = true;
    }
  }
}

function buildStreetlights(scene, lights) {
  const poleGeo = buildStreetPoleGeo();
  const headGeo = buildStreetHeadGeo();
  const metal = new THREE.MeshStandardMaterial({ color: 0x33363c, metalness: 0.75, roughness: 0.42 });
  const headMat = new THREE.MeshStandardMaterial({
    color: 0x22252a, emissive: 0xffcf8f, emissiveIntensity: 0, roughness: 0.5, metalness: 0.2,
    fog: false,
  });

  const transforms = [];
  for (const l of lights) {
    const y = heightAt(worldRef, l.x, l.z);
    transforms.push({ matrix: txAt(l.x, y, l.z, l.rotY, 1) });
  }
  inst(scene, poleGeo, metal, transforms);
  state.streetHead = inst(scene, headGeo, headMat, transforms);

  // warm ground light pool (bigger)
  const glowTex = state.glowTex;
  const glowBigMat = new THREE.MeshBasicMaterial({
    map: glowTex, transparent: true, opacity: 0, fog: false,
    blending: THREE.AdditiveBlending, depthWrite: false, color: 0xffb060,
  });
  const poolGeo = placed(new THREE.PlaneGeometry(20, 20), 0, 0, 0, -Math.PI / 2);
  const poolTransforms = [];
  for (const l of lights) {
    const y = heightAt(worldRef, l.x, l.z);
    poolTransforms.push({ matrix: txAt(l.x, y + 0.12, l.z, 0, 1) });
  }
  state.glowBig = inst(scene, poolGeo, glowBigMat, poolTransforms);

  state.headMats.push({ mat: headMat, glow: glowBigMat });
}

function buildLampPosts(scene, lamps) {
  if (!lamps.length) return;
  const bodyGeo = buildLampBodyGeo();
  const headGeo = buildLampHeadGeo();
  const metal = new THREE.MeshStandardMaterial({ color: 0x2c3035, metalness: 0.7, roughness: 0.5 });
  const headMat = new THREE.MeshStandardMaterial({
    color: 0x23262b, emissive: 0xffdfa6, emissiveIntensity: 0, roughness: 0.5, metalness: 0.1,
    fog: false,
  });

  const transforms = [];
  for (const l of lamps) {
    const y = heightAt(worldRef, l.x, l.z);
    transforms.push({ matrix: txAt(l.x, y, l.z, state.rng() * Math.PI * 2, 1) });
  }
  inst(scene, bodyGeo, metal, transforms);
  state.lampHead = inst(scene, headGeo, headMat, transforms);

  const glowSmallMat = new THREE.MeshBasicMaterial({
    map: state.glowTex, transparent: true, opacity: 0, fog: false,
    blending: THREE.AdditiveBlending, depthWrite: false, color: 0xffc070,
  });
  const poolGeo = placed(new THREE.PlaneGeometry(12, 12), 0, 0, 0, -Math.PI / 2);
  const poolTransforms = [];
  for (const l of lamps) {
    const y = heightAt(worldRef, l.x, l.z);
    poolTransforms.push({ matrix: txAt(l.x, y + 0.12, l.z, 0, 1) });
  }
  state.glowSmall = inst(scene, poolGeo, glowSmallMat, poolTransforms);
  state.headMats.push({ mat: headMat, glow: glowSmallMat });
}

function buildBenches(scene, benches) {
  if (!benches.length) return;
  const geo = buildBenchGeo();
  const mat = new THREE.MeshStandardMaterial({
    map: state.barkTex, color: 0xb49a72, roughness: 0.9, metalness: 0.0,
  });
  const transforms = [];
  for (const b of benches) {
    const y = heightAt(worldRef, b.x, b.z);
    transforms.push({ matrix: txAt(b.x, y + 0.02, b.z, b.rotY || 0, 1) });
  }
  inst(scene, geo, mat, transforms);
}

function buildHydrants(scene, hydrants) {
  if (!hydrants.length) return;
  const geo = buildHydrantGeo();
  const mat = new THREE.MeshStandardMaterial({ color: 0xc2262b, roughness: 0.5, metalness: 0.25 });
  const transforms = [];
  for (const h of hydrants) {
    const y = heightAt(worldRef, h.x, h.z);
    transforms.push({ matrix: txAt(h.x, y + 0.01, h.z, h.rotY || 0, 1) });
  }
  inst(scene, geo, mat, transforms);
}

function buildBusStops(scene, buses) {
  if (!buses.length) return;
  const frameGeo = buildBusFrameGeo();
  const glassGeo = buildBusGlassGeo();
  const frameMat = new THREE.MeshStandardMaterial({ color: 0x39414a, metalness: 0.6, roughness: 0.5 });
  const glassMat = new THREE.MeshStandardMaterial({
    color: 0xbfe4f2, transparent: true, opacity: 0.32, roughness: 0.05, metalness: 0.1,
  });
  const transforms = [];
  for (const b of buses) {
    const y = heightAt(worldRef, b.x, b.z);
    transforms.push({ matrix: txAt(b.x, y + 0.02, b.z, b.rotY || 0, 1) });
  }
  inst(scene, frameGeo, frameMat, transforms);
  inst(scene, glassGeo, glassMat, transforms);
}

function buildFences(scene, parkSpots) {
  if (!parkSpots.length) return;
  const geo = buildFenceGeo();
  const mat = new THREE.MeshStandardMaterial({ color: 0x5a6b50, roughness: 0.85, metalness: 0.1 });
  const transforms = [];
  for (const p of parkSpots) {
    const cy = heightAt(worldRef, p.x, p.z);
    const N = 7;                       // ring of panels around the green
    for (let k = 0; k < N; k++) {
      const a = (k / N) * Math.PI * 2 + state.rng() * 0.4;
      const r = 3.6;
      const x = p.x + Math.cos(a) * r, z = p.z + Math.sin(a) * r;
      const y = heightAt(worldRef, x, z);
      transforms.push({ matrix: txAt(x, y + 0.02, z, a + Math.PI / 2, 1) });
    }
  }
  inst(scene, geo, mat, transforms);
}

// ---------------------------------------------------------------------------
// Day/night factor (reuse the environment's solar-elevation model)
// ---------------------------------------------------------------------------
function nightFactor(todSec) {
  const noon = 12 * 3600;
  const h = (todSec - noon) / 3600;
  const half = 6.5;
  const x = Math.min(Math.abs(h) / half, 1.4);
  const elev = (Math.PI / 2) * Math.cos(x * (Math.PI / 2));
  return clamp((-elev - 0.02) / 0.12, 0, 1);   // 0 by day -> 1 deep night
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
let worldRef = null;

export function init(world) {
  if (state.built) return;
  if (!world || !world.scene) { state.built = true; return; }
  worldRef = world;

  const seed = (world.meta && typeof world.meta.seed === 'number') ? world.meta.seed : 1337;
  state.rng = localRNG(seed);

  // shared atlas textures
  state.leafTex = buildLeafTexture(seed);
  state.barkTex = buildBarkTexture(seed);
  state.glowTex = buildGlowTexture(seed);

  state.group = new THREE.Group();
  world.scene.add(state.group);

  try {
    placeEverything(world);
  } catch (e) {
    // isolation: never take down the app
    console.error('[props] placement failed', e);
  }

  applyNight(world);
  state.built = true;
}

export function update(dtSec, world) {
  if (!state.built || !world || !world.meta) return;
  const tod = typeof world.meta.timeOfDaySec === 'number' ? world.meta.timeOfDaySec : 9 * 3600;
  applyNight(world, tod);
}

function applyNight(world, tod) {
  if (!state.headMats.length) return;
  const f = nightFactor(tod != null ? tod : ((world && world.meta && world.meta.timeOfDaySec) || 9 * 3600));
  for (const hm of state.headMats) {
    if (hm.mat) hm.mat.emissiveIntensity = f * 12;
    if (hm.glow) hm.glow.opacity = f * 0.9;
  }
}

export function showcase(container) {
  const c = state.counts || {};
  try {
    if (container) {
      container.innerHTML =
        `<div style="padding:14px;font-family:sans-serif;color:#d7f2c9;background:#0d1a10">
          <b>Props</b> — instanced street furniture &amp; vegetation.<br>
          trees ${c.trees ?? 0} · streetlights ${c.streetLights ?? 0} ·
          benches ${c.benches ?? 0} · hydrants ${c.hydrants ?? 0} ·
          bus stops ${c.busStops ?? 0} · lamp posts ${c.lampPosts ?? 0} ·
          parks ${c.parks ?? 0}.<br>
          Night: emissive heads + warm ground light pools.
        </div>`;
    }
  } catch (e) { /* ignore */ }
}
