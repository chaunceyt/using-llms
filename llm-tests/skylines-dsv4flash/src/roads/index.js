// ============================================================================
// roads module — Skylines
// Deterministic road network with believable asphalt PBR surface + markings.
//
// Owns ONLY src/roads/. Everything here is a pure function of world.meta.seed.
// Draw calls: 1 asphalt mesh + 1 markings mesh (+ guarded fallback ground/light).
// ============================================================================
import * as THREE from 'three';
import { bus, world } from '../core/index.js';

export const id = 'roads';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;

function dist2(ax, az, bx, bz) {
  const dx = bx - ax, dz = bz - az;
  return Math.sqrt(dx * dx + dz * dz);
}

// deterministic per-pixel hash noise (independent of world.rng draw order)
function hnoise(x, y, s) {
  let n = ((x * 374761393 + y * 668265263 + (s | 0) * 2246822519) >>> 0);
  n = Math.imul(n ^ (n >>> 13), 1274126177);
  return (((n ^ (n >>> 16)) >>> 0) / 4294967296);
}

// ---------------------------------------------------------------------------
// Procedural PBR textures for asphalt (canvas, deterministic from seed)
// ---------------------------------------------------------------------------
function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return { c, ctx: c.getContext('2d') };
}

function buildTextures(seed) {
  const N = 256;
  // --- height field for normal derivation ---
  const hf = new Float32Array(N * N);
  const layer = (px, py, freq, amp) => hnoise((px / N) * freq, (py / N) * freq, seed + freq * 91) * amp;
  let minv = Infinity, maxv = -Infinity;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let v = 0;
      v += layer(x, y, 3, 1.0);
      v += layer(x, y, 7, 0.55);
      v += layer(x, y, 18, 0.30);
      v += (hnoise(x * 3, y * 3, seed) - 0.5) * 0.15;
      const h = Math.pow(v, 1.4);
      hf[y * N + x] = h;
      if (h < minv) minv = h; if (h > maxv) maxv = h;
    }
  }

  // --- albedo ---
  const alb = makeCanvas(N, N);
  {
    const img = alb.ctx.createImageData(N, N);
    for (let i = 0; i < N * N; i++) {
      const n = clamp((hf[i] - minv) / (maxv - minv + 1e-6), 0, 1);
      // asphalt is a cool dark grey with subtle grain
      let g = 44 + n * 26;
      g += (hnoise(i % N, (i / N) | 0, seed + 7) - 0.5) * 6;
      const r = g * 1.03, b = g * 1.06;
      img.data[i * 4] = clamp(r, 0, 255);
      img.data[i * 4 + 1] = clamp(g, 0, 255);
      img.data[i * 4 + 2] = clamp(b, 0, 255);
      img.data[i * 4 + 3] = 255;
    }
    alb.ctx.putImageData(img, 0, 0);
  }

  // --- normal (tangent space) from height gradient ---
  const nrm = makeCanvas(N, N);
  {
    const img = nrm.ctx.createImageData(N, N);
    const strength = 1.6;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const s = 2.5;
        const hL = hf[y * N + ((x - 1 + N) % N)];
        const hR = hf[y * N + ((x + 1) % N)];
        const hD = hf[((y - 1 + N) % N) * N + x];
        const hU = hf[((y + 1) % N) * N + x];
        const dx = (hR - hL) / s;
        const dz = (hU - hD) / s;
        const nx = -dx * strength, ny = 1.0, nz = -dz * strength;
        const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
        img.data[(y * N + x) * 4] = clamp((nx / l * 0.5 + 0.5) * 255, 0, 255);
        img.data[(y * N + x) * 4 + 1] = clamp((ny / l * 0.5 + 0.5) * 255, 0, 255);
        img.data[(y * N + x) * 4 + 2] = clamp((nz / l * 0.5 + 0.5) * 255, 0, 255);
        img.data[(y * N + x) * 4 + 3] = 255;
      }
    }
    nrm.ctx.putImageData(img, 0, 0);
  }

  // --- roughness (green-ish channel; mostly matte with variation) ---
  const rgh = makeCanvas(N, N);
  {
    const img = rgh.ctx.createImageData(N, N);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        let g = 210 + (hnoise(x * 5, y * 5, seed + 3) - 0.5) * 34;
        g += (1 - clamp((hf[y * N + x] - minv) / (maxv - minv + 1e-6), 0, 1)) * 12;
        const idx = (y * N + x) * 4;
        img.data[idx] = g;     // R unused
        img.data[idx + 1] = g; // G = roughness
        img.data[idx + 2] = 20;// B unused (metalness ~0)
        img.data[idx + 3] = 255;
      }
    }
    rgh.ctx.putImageData(img, 0, 0);
  }

  // --- AO: mostly open, subtle darkening in micro-cracks ---
  const ao = makeCanvas(N, N);
  {
    const img = ao.ctx.createImageData(N, N);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const n = clamp((hf[y * N + x] - minv) / (maxv - minv + 1e-6), 0, 1);
        const v = 235 - n * 26;
        const idx = (y * N + x) * 4;
        img.data[idx] = v; img.data[idx + 1] = v; img.data[idx + 2] = v; img.data[idx + 3] = 255;
      }
    }
    ao.ctx.putImageData(img, 0, 0);
  }

  const mkTex = (canvas, srgb) => {
    const t = new THREE.CanvasTexture(canvas.c);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace; else t.colorSpace = THREE.NoColorSpace;
    return t;
  };

  return {
    albedo: mkTex(alb, true),
    normal: mkTex(nrm, false),
    roughness: mkTex(rgh, false),
    ao: mkTex(ao, false),
  };
}

// ---------------------------------------------------------------------------
// Index-welded geometry builder (single merged buffer -> low draw calls)
// ---------------------------------------------------------------------------
class GeoBuilder {
  constructor() {
    this.pos = []; this.norm = []; this.uv = [];
    this.idx = []; this.map = new Map();
  }
  addVertex(x, y, z, u, v) {
    const key = (Math.round(x * 50)) + ',' + (Math.round(z * 50)) + ',' + (Math.round(y * 50));
    let i = this.map.get(key);
    if (i === undefined) {
      i = this.pos.length / 3;
      this.pos.push(x, y, z);
      this.norm.push(0, 1, 0);
      this.uv.push(u, v);
      this.map.set(key, i);
    }
    return i;
  }
  quad(A, B, C, D) {
    const a = this.addVertex(...A), b = this.addVertex(...B), c = this.addVertex(...C), d = this.addVertex(...D);
    this.idx.push(a, b, c, a, c, d);
  }
  toGeometry() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.norm, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
    // second uv channel for aoMap
    g.setAttribute('uv2', new THREE.Float32BufferAttribute(this.uv.slice(), 2));
    g.setIndex(this.idx);
    g.computeBoundingSphere();
    return g;
  }
}

// ---------------------------------------------------------------------------
// Road module state + public API
// ---------------------------------------------------------------------------
const TEX_SCALE = 5;          // metres per texture tile (world-space UV)
const MARK_ALT = 0.025;       // markings float just above asphalt
const JUNCTION_GAP = 1.2;     // markings stop this far from a junction

const state = {
  built: false,
  corridors: [],      // { id, points:[[x,z]...], width, type }
  segments: [],       // flattened { ci, k, ax,az,bx,bz, width } for surfaceAt
  texGroup: null,
  roadMesh: null,
  markMesh: null,
  fallbackGround: null,
};

function heightAt(world, x, z) {
  try {
    const f = world && world.terrain && world.terrain.heightAt;
    if (typeof f === 'function') { const v = f(x, z); return isFinite(v) ? v : 0; }
  } catch (e) { /* terrain absent/err -> plane */ }
  return 0;
}

// perpendicular to a-b in the XZ plane
function perpOf(ax, az, bx, bz) {
  const dx = bx - ax, dz = bz - az;
  const l = Math.sqrt(dx * dx + dz * dz) || 1;
  return { x: dz / l, z: -dx / l };
}

function pointAt(points, d) {
  // points: [[x,z]...]; returns [x,z] at arc distance d
  let cum = 0;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i], b = points[i + 1];
    const len = dist2(a[0], a[1], b[0], b[1]);
    if (cum + len >= d) {
      const t = clamp((d - cum) / (len || 1), 0, 1);
      return [lerp(a[0], b[0], t), lerp(a[1], b[1], t)];
    }
    cum += len;
  }
  const last = points[points.length - 1];
  return [last[0], last[1]];
}

// ---------------------------------------------------------------------------
// Layout: pure function of the seed. Grid + diagonals + a curving arterial.
// ---------------------------------------------------------------------------
function layoutCity(rng) {
  const roads = [];
  const mainXs = [-240, 0, 240];
  // vertical avenues
  for (const x of [-480, -360, -240, -120, 0, 120, 240, 360, 480]) {
    const isMain = mainXs.includes(x);
    roads.push({
      points: [[x, -500], [x, 500]],
      width: isMain ? 20 : (rng() < 0.5 ? 15 : 12),
      type: isMain ? 'avenue' : 'collector',
    });
  }
  // horizontal avenues
  for (const z of [-480, -360, -240, -120, 0, 120, 240, 360, 480]) {
    const isMain = mainXs.includes(z);
    roads.push({
      points: [[-500, z], [500, z]],
      width: isMain ? 20 : (rng() < 0.5 ? 15 : 12),
      type: isMain ? 'avenue' : 'collector',
    });
  }
  // diagonal main avenue (~32 deg)
  roads.push({ points: [[-520, -300], [470, 330]], width: 22, type: 'main' });
  // secondary diagonal (~ -40 deg)
  roads.push({ points: [[-440, 360], [420, -400]], width: 16, type: 'collector' });
  // curving arterial (gentle S) — organic counterpoint to the grid
  const curve = [];
  const N = 22;
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    const x = -480 + 960 * t;
    const z = 300 * Math.sin(t * Math.PI) - 440 * (1 - 2 * t);
    curve.push([x, z]);
  }
  roads.push({ points: curve, width: 18, type: 'avenue' });

  // a few short local connector streets off the main grid for texture
  const extra = rng() < 0.6;
  if (extra) {
    const xs = [-180, 60, 300].map(x => x + (rng() - 0.5) * 8);
    for (const x of xs) {
      roads.push({ points: [[x, -500], [x, 500]], width: 12, type: 'street' });
    }
  }
  return roads;
}

// ---------------------------------------------------------------------------
// Compute crossing distances along each corridor
// ---------------------------------------------------------------------------
function computeCrossings(corridors) {
  // build global segments
  const segs = [];
  corridors.forEach((c, ci) => {
    let cum = 0;
    for (let k = 0; k < c.points.length - 1; k++) {
      const a = c.points[k], b = c.points[k + 1];
      segs.push({ ci, k, a, b, dStart: cum, len: dist2(a[0], a[1], b[0], b[1]) });
      cum += segs[segs.length - 1].len;
    }
  });

  // per-corridor sorted list of crossing distances
  const crossings = corridors.map(() => []);
  for (let i = 0; i < segs.length; i++) {
    const A = segs[i];
    for (let j = i + 1; j < segs.length; j++) {
      const B = segs[j];
      if (A.ci === B.ci) continue;
      // skip touching end points (adjacent roads)
      const denom = (B.b[0] - B.a[0]) * (A.b[1] - A.a[1]) - (B.b[1] - B.a[1]) * (A.b[0] - A.a[0]);
      if (Math.abs(denom) < 1e-9) continue;
      const t = ((B.a[0] - A.a[0]) * (A.b[1] - A.a[1]) - (B.a[1] - A.a[1]) * (A.b[0] - A.a[0])) / denom;
      if (t < 1e-4 || t > 1 - 1e-4) continue;
      const u = ((B.a[0] - A.a[0]) * (B.b[1] - B.a[1]) - (B.a[1] - A.a[1]) * (B.b[0] - B.a[0])) / denom;
      if (u < 1e-4 || u > 1 - 1e-4) continue;
      crossings[A.ci].push(A.dStart + t * A.len);
      crossings[B.ci].push(B.dStart + u * B.len);
    }
  }
  // sort + dedupe per corridor
  return crossings.map(c => {
    c.sort((a, b) => a - b);
    const out = [];
    for (const v of c) if (!out.length || v - out[out.length - 1] > 0.5) out.push(v);
    return out;
  });
}

// ---------------------------------------------------------------------------
// Markings
// ---------------------------------------------------------------------------
function laneOffsets(width) {
  const n = clamp(Math.round(width / 4.8), 2, 6);
  const edgeOff = width / 2 - 0.18;
  const bds = [];
  for (let i = 1; i < n; i++) bds.push(-width / 2 + i * (width / n));
  return { n, edgeOff, bds };
}

function emitStrip(g, path, lateralOff, halfThick, world) {
  for (let k = 0; k < path.length - 1; k++) {
    const a = path[k], b = path[k + 1];
    const p = perpOf(a[0], a[1], b[0], b[1]);
    const c1 = [a[0] + p.x * (lateralOff - halfThick), heightAt(world, a[0], a[1]) + MARK_ALT, a[1] + p.z * (lateralOff - halfThick), 0, 0];
    const c2 = [a[0] + p.x * (lateralOff + halfThick), heightAt(world, a[0], a[1]) + MARK_ALT, a[1] + p.z * (lateralOff + halfThick), 0, 0];
    const c3 = [b[0] + p.x * (lateralOff + halfThick), heightAt(world, b[0], b[1]) + MARK_ALT, b[1] + p.z * (lateralOff + halfThick), 0, 0];
    const c4 = [b[0] + p.x * (lateralOff - halfThick), heightAt(world, b[0], b[1]) + MARK_ALT, b[1] + p.z * (lateralOff - halfThick), 0, 0];
    g.quad(c1, c2, c3, c4);
  }
}

function samplePath(points, s, e) {
  const out = [];
  let cum = 0;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i], b = points[i + 1];
    const len = dist2(a[0], a[1], b[0], b[1]);
    if (cum + len < s) { cum += len; continue; }
    if (cum > e) break;
    const segStart = cum, segEnd = cum + len;
    out.push(pointAt(points, Math.max(segStart, s)));
    let t = 0.7;
    while (segStart + t < Math.min(segEnd, e)) { out.push(pointAt(points, segStart + t)); t += 0.7; }
    out.push(pointAt(points, Math.min(segEnd, e)));
    cum = segEnd;
  }
  return out;
}

function buildMarkings(g, corridor, world) {
  const width = corridor.width;
  const { edgeOff, bds } = laneOffsets(width);
  const L = corridor.crossD.length ? corridor.crossD[corridor.crossD.length - 1] + 10 : (() => {
    let c = 0; for (let i = 0; i < corridor.points.length - 1; i++) c += dist2(corridor.points[i][0], corridor.points[i][1], corridor.points[i + 1][0], corridor.points[i + 1][1]);
    return c;
  })();

  // spans of road that are NOT near a junction
  const spans = [];
  const d = corridor.crossD;
  let prev = 0;
  for (const v of d) {
    if (v - prev > JUNCTION_GAP * 2 + 0.5) spans.push([prev + JUNCTION_GAP, v - JUNCTION_GAP]);
    prev = v;
  }
  if (L - prev > JUNCTION_GAP * 2 + 0.5) spans.push([prev + JUNCTION_GAP, L - JUNCTION_GAP]);

  // central boundary handling
  const n = laneOffsets(width).n;
  const centerOff = n % 2 === 0 ? null : bds[Math.floor(bds.length / 2)];

  for (const [s, e] of spans) {
    if (e <= s) continue;
    // edge solid lines
    const lEdge = samplePath(corridor.points, s, e);
    emitStrip(g, lEdge, +edgeOff, 0.09, world);
    emitStrip(g, lEdge, -edgeOff, 0.09, world);

    // interior dividers
    for (const off of bds) {
      const isCenter = n % 2 === 1 && Math.abs(off - centerOff) < 1e-6;
      if (isCenter) {
        // dashed white centre line for odd lanes
        emitDashes(g, corridor.points, s, e, off, world);
      } else if (n % 2 === 0 && Math.abs(off) < 1e-6) {
        // even lanes -> double solid yellow at the very middle
        const p = samplePath(corridor.points, s, e);
        emitStrip(g, p, -0.35, 0.075, world);
        emitStrip(g, p, +0.35, 0.075, world);
      } else {
        emitDashes(g, corridor.points, s, e, off, world);
      }
    }
  }
}

function emitDashes(g, points, s, e, lateralOff, world) {
  const dashLen = 2.6, gap = 3.1;
  let pos = s;
  while (pos < e) {
    const end = Math.min(pos + dashLen, e);
    const a = pointAt(points, pos), b = pointAt(points, end);
    const p = perpOf(a[0], a[1], b[0], b[1]);
    const hw = 0.09;
    const c1 = [a[0] + p.x * (lateralOff - hw), heightAt(world, a[0], a[1]) + MARK_ALT, a[1] + p.z * (lateralOff - hw), 0, 0];
    const c2 = [a[0] + p.x * (lateralOff + hw), heightAt(world, a[0], a[1]) + MARK_ALT, a[1] + p.z * (lateralOff + hw), 0, 0];
    const c3 = [b[0] + p.x * (lateralOff + hw), heightAt(world, b[0], b[1]) + MARK_ALT, b[1] + p.z * (lateralOff + hw), 0, 0];
    const c4 = [b[0] + p.x * (lateralOff - hw), heightAt(world, b[0], b[1]) + MARK_ALT, b[1] + p.z * (lateralOff - hw), 0, 0];
    g.quad(c1, c2, c3, c4);
    pos += dashLen + gap;
  }
}

// ---------------------------------------------------------------------------
// Geometry assembly
// ---------------------------------------------------------------------------
function buildSurface(world) {
  const asphalt = new GeoBuilder();
  const marks = new GeoBuilder();

  // full-width ribbons (overlapping at junctions is fine — shared world-space UVs)
  for (const c of state.corridors) {
    for (let k = 0; k < c.points.length - 1; k++) {
      const a = c.points[k], b = c.points[k + 1];
      const hw = c.width / 2;
      const p = perpOf(a[0], a[1], b[0], b[1]);
      const ya = heightAt(world, a[0], a[1]), yb = heightAt(world, b[0], b[1]);
      const A = [a[0] + p.x * hw, ya, a[1] + p.z * hw, a[0] / TEX_SCALE, a[1] / TEX_SCALE];
      const B = [a[0] - p.x * hw, ya, a[1] - p.z * hw, a[0] / TEX_SCALE, a[1] / TEX_SCALE];
      const C = [b[0] - p.x * hw, yb, b[1] - p.z * hw, b[0] / TEX_SCALE, b[1] / TEX_SCALE];
      const D = [b[0] + p.x * hw, yb, b[1] + p.z * hw, b[0] / TEX_SCALE, b[1] / TEX_SCALE];
      asphalt.quad(A, B, C, D);
    }
  }

  // markings per corridor (clipped away from junctions)
  for (const c of state.corridors) buildMarkings(marks, c, world);

  return { asphalt: asphalt.toGeometry(), marks: marks.toGeometry() };
}

// ---------------------------------------------------------------------------
// Guards for missing terrain/environment so roads are still visible
// ---------------------------------------------------------------------------
function ensureFallbacks(world) {
  const scene = world.scene;

  // ground plane fallback if no module provides one yet
  let hasGround = false;
  scene.traverse(o => { if (o.isMesh && o.name === '__skylines_ground') hasGround = true; });
  if (!hasGround && !(world.terrain && world.terrain.mesh)) {
    const size = (world.terrain && world.terrain.size) || 2048;
    const geo = new THREE.PlaneGeometry(size, size);
    geo.rotateX(-Math.PI / 2);
    const mat = new THREE.MeshStandardMaterial({ color: 0x4b5446, roughness: 1, metalness: 0 });
    const ground = new THREE.Mesh(geo, mat);
    ground.name = '__skylines_ground';
    ground.position.y = -0.02;
    ground.receiveShadow = true;
    scene.add(ground);
    state.fallbackGround = ground;
  }

  // lighting fallback if environment hasn't added lights yet
  let hasLight = false;
  scene.traverse(o => { if (o.isLight) hasLight = true; });
  if (!hasLight) {
    const hemi = new THREE.HemisphereLight(0xbdd0e6, 0x33382e, 0.85);
    const sun = new THREE.DirectionalLight(0xfff2d9, 1.6);
    sun.position.set(400, 500, 300);
    scene.add(hemi, sun);
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
export function init(world) {
  if (state.built) return;
  ensureFallbacks(world);
  buildNetwork(world);
}

export function buildNetwork(world, layout) {
  const scene = world.scene;

  // deterministic RNG derived from the world seed (independent of other draws)
  const localSeed = ((world.meta.seed ^ 0x9e3779b9) >>> 0);
  const rng = mulberry(localSeed);
  const corridors = layout || layoutCity(rng);

  state.corridors = corridors.map((c, i) => ({ id: `road${i}`, ...c }));
  state.built = true;

  // crossings for clean junctions
  const crossD = computeCrossings(state.corridors);
  state.corridors.forEach((c, i) => { c.crossD = crossD[i]; });

  // flattened segments + world.roads + events
  state.segments.length = 0;
  let segId = 0;
  for (const c of state.corridors) {
    for (let k = 0; k < c.points.length - 1; k++) {
      const a = c.points[k], b = c.points[k + 1];
      const seg = { id: `${c.id}_${segId++}`, from: [a[0], a[1]], to: [b[0], b[1]], width: c.width, type: c.type };
      world.roads.push(seg);
      state.segments.push({ ax: a[0], az: a[1], bx: b[0], bz: b[1], len: dist2(a[0], a[1], b[0], b[1]), width: c.width });
      bus.emit('road:add', { segment: seg });
    }
  }

  // textures + materials (reused across everything -> 1 asphalt + 1 markings call)
  if (!state.texGroup) state.texGroup = buildTextures(localSeed);
  const tg = state.texGroup;
  for (const t of [tg.albedo, tg.normal, tg.roughness, tg.ao]) {
    t.repeat.set(1, 1); t.anisotropy = 4; t.needsUpdate = true;
  }
  const roadMat = new THREE.MeshStandardMaterial({
    map: tg.albedo,
    normalMap: tg.normal,
    roughnessMap: tg.roughness,
    aoMap: tg.ao,
    roughness: 0.9, metalness: 0.0,
    side: THREE.DoubleSide,
  });
  const markMat = new THREE.MeshStandardMaterial({
    color: 0xf3f4ed, roughness: 0.7, metalness: 0.0,
    side: THREE.DoubleSide,
  });

  // build merged geometry
  const { asphalt, marks } = buildSurface(world);

  if (state.roadMesh) scene.remove(state.roadMesh);
  if (state.markMesh) scene.remove(state.markMesh);
  state.roadMesh = new THREE.Mesh(asphalt, roadMat);
  state.roadMesh.name = 'roads';
  state.roadMesh.receiveShadow = true;
  state.markMesh = new THREE.Mesh(marks, markMat);
  state.markMesh.name = 'road_markings';
  scene.add(state.roadMesh, state.markMesh);

  return { corridors: state.corridors, count: world.roads.length };
}

export function addRoad(from, to, opts = {}) {
  const w = opts.width || 14;
  const c = { points: [[from[0], from[2] ?? from[1]], [to[0], to[2] ?? to[1]]], width: w, type: opts.type || 'collector' };
  state.corridors.push(c);
  // recompute crossings + rebuild (simplest correct path for now)
  const crossD = computeCrossings(state.corridors);
  state.corridors.forEach((cc, i) => { cc.crossD = crossD[i]; });
  rebuildMeshes(world);
  return c.id;
}

export function removeRoad(id) {
  state.corridors = state.corridors.filter(c => c.id !== id);
  const segsBefore = world.roads.length;
  world.roads = world.roads.filter(s => !s.id.startsWith(id + '_'));
  if (world.roads.length !== segsBefore) bus.emit('road:remove', { id });
  rebuildMeshes(world);
}

function rebuildMeshes(world) {
  // drop old mesh state so a fresh network regenerates
  const scene = world.scene;
  if (state.roadMesh) { scene.remove(state.roadMesh); state.roadMesh = null; }
  if (state.markMesh) { scene.remove(state.markMesh); state.markMesh = null; }
  // re-run buildSurface into existing materials
  const tg = state.texGroup;
  const roadMat = new THREE.MeshStandardMaterial({
    map: tg.albedo, normalMap: tg.normal, roughnessMap: tg.roughness, aoMap: tg.ao,
    roughness: 0.9, metalness: 0.0, side: THREE.DoubleSide,
  });
  const markMat = new THREE.MeshStandardMaterial({ color: 0xf3f4ed, roughness: 0.7, metalness: 0.0, side: THREE.DoubleSide });
  const { asphalt, marks } = buildSurface(world);
  state.roadMesh = new THREE.Mesh(asphalt, roadMat); state.roadMesh.name = 'roads';
  state.markMesh = new THREE.Mesh(marks, markMat); state.markMesh.name = 'road_markings';
  scene.add(state.roadMesh, state.markMesh);
}

export function surfaceAt(x, z) {
  let best = null, bd = Infinity;
  for (const s of state.segments) {
    // projection onto segment
    const vx = s.bx - s.ax, vz = s.bz - s.az;
    const l2 = vx * vx + vz * vz || 1;
    let t = ((x - s.ax) * vx + (z - s.az) * vz) / l2;
    t = clamp(t, 0, 1);
    const cx = s.ax + t * vx, cz = s.az + t * vz;
    const d = dist2(x, z, cx, cz);
    if (d < bd) { bd = d; best = { s, t, cx, cz }; }
  }
  if (!best || bd > best.s.width / 2) return { onRoad: false, lane: null, dir: null };
  const dl = Math.sqrt(best.s.len * best.s.len);
  const dirX = (best.s.bx - best.s.ax) / (dl || 1), dirZ = (best.s.bz - best.s.az) / (dl || 1);
  // signed lateral offset relative to direction
  const side = ((x - best.cx) * dirZ - (z - best.cz) * (-dirX)); // simple sign heuristic
  const lane = Math.abs(side) < best.s.width / 8 ? 'center' : (side > 0 ? 'right' : 'left');
  return { onRoad: true, lane, dir: { x: dirX, z: dirZ }, width: best.s.width };
}

export function update(dtSec, world) {
  // static network; nothing per-frame.
}

export function showcase(container) {
  if (!state.built) init(world);
  const d = document.createElement('div');
  d.style.cssText = 'color:#cfd6d9;font:12px system-ui;padding:8px';
  d.textContent = `roads module — ${world.roads.length} segments, grid + diagonals + arterial`;
  if (container) container.appendChild(d);
}

// local seeded rng for texture/layout isolation (mulberry32)
function mulberry(a0) {
  let a = a0 >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
