// props — the city's living cover: four species of instanced trees, streetlights
// lining every road (heads glow warm at night AND cast a fake additive light
// pool on the asphalt), hedge bushes, bollards and park benches.
//
// Draw calls: 10 — one InstancedMesh per asset class
//   [0] broadleaf trees A (tapered trunk + branches + 5 displaced canopy blobs)
//   [1] broadleaf trees B (wide, low "spread" form, 5 displaced blobs)
//   [2] conifer trees     (trunk + 4 stacked flattened displaced cones)
//   [3] umbrella trees    (tall trunk + high branches + wide flat canopy ring —
//       also used as the aligned street-tree rows)
//   [4] streetlight poles (base + pole + arm + head housing, merged, metallic)
//   [5] streetlight glow heads (unlit lens ellipsoids; material colour ramps
//       0 by day -> bright warm by night)
//   [6] streetlight light pools (flat additive quads under each head, radial
//       gradient texture; material colour ramps with the clock — the "light
//       spill" on the asphalt, no real point lights on SwiftShader)
//   [7] bushes / low hedges (3 displaced blobs; hedges are stretched instances)
//   [8] bollards (plaza ring + reflective band)
//   [9] benches (wood seat + back, metal legs, placed in park pockets facing roads)
//
// Trees are varied per instance: scale 0.6-1.8, anisotropic scale (wide/low vs
// narrow/tall silhouettes), random yaw, and green tint jitter via instanceColor
// (some yellowish, some deep green) so a grove never reads as clones.
//
// Placement (deterministic, ctx.rng.fork('props') only):
//   trees  — jittered 7 m grid over the plain + lower hills (r<192,
//            h in (-1, 40], slope <= ~0.6), kept 6 m clear of road
//            centerlines and off building footprints (+2 m margin).
//            Low-freq fBm carves "park" pockets that densify 4x4 into
//            forest (2.6 m spacing), so groves read as groves; the rest of
//            the green gets 2x2 lawn scatter. Per-tree shape/tint is a pure
//            hash of the spot position (stable, stream-independent).
//   street trees — one aligned row per road edge, ~15 m spacing, 5.9 m off the
//            centerline, yawed along the road, skipped at junctions and near
//            lamps.
//   lights — every road edge, ~21 m spacing, 5.5 m off the centerline
//            (just outside the 10 m carriageway), alternating sides,
//            arm pointing back at the road.
//   bushes — park pockets only, 1-2 per spot (~1/3 stretched into low hedges).
//   bollards — sidewalks of the core's arterials (6.5 m off centerline,
//            a handful per long edge, clear of lamps).
//   benches  — park pockets, 6-35 m from a road, facing it.
//
// Public API: addProp(type, pos) ['tree'|'light'], setCount(type, n),
// world.props = { trees, lights }. Emits 'props:ready' {trees, lights}.
// update() touches a handful of material uniforms per frame — no per-instance
// work, no allocations (module-scope scratch).
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const smooth01 = (x) => { const t = clamp(x, 0, 1); return t * t * (3 - 2 * t); };

// ---- deterministic 2-D value-noise fBm (pure function of seed — no RNG) ----
function makeFbm(seed) {
  const hash = (x, y) => {
    let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(seed, 974634799)) >>> 0;
    h = (h ^ (h >>> 13)) >>> 0; h = Math.imul(h, 0x5bd1e995) >>> 0; h ^= h >>> 15;
    return (h >>> 0) / 4294967296;
  };
  const noise2 = (x, y) => {
    const ix = Math.floor(x), iy = Math.floor(y);
    const fx = x - ix, fy = y - iy;
    const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
    const a = hash(ix, iy), b = hash(ix + 1, iy), c = hash(ix, iy + 1), d = hash(ix + 1, iy + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  };
  // two octaves -> smooth park-pocket mask in 0..1
  return (x, y) => noise2(x, y) * 0.65 + noise2(x * 2.03 + 17.7, y * 2.03 - 9.1) * 0.35;
}

// Deterministic per-vertex hash (for organic silhouette displacement + colour
// jitter). Pure integer math — identical on every run.
function _hash3(x, y, z, s) {
  const ix = Math.round(x * 31.7) | 0, iy = Math.round(y * 29.3) | 0, iz = Math.round(z * 37.1) | 0;
  let h = (Math.imul(ix, 374761393) ^ Math.imul(iy, 668265263) ^ Math.imul(iz, 1440662683) ^ Math.imul(s, 974634799)) >>> 0;
  h = (h ^ (h >>> 13)) >>> 0; h = Math.imul(h, 0x5bd1e995) >>> 0; h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

// Radial vertex displacement from the geometry origin — turns perfect
// icospheres/cones into lumpy, hand-placed-looking foliage mass.
function _displace(g, seed, amp) {
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    const k = 1 + amp * (2 * _hash3(x, y, z, seed) - 1);
    p.setXYZ(i, x * k, y * k, z * k);
  }
  return g;
}

// Paint a flat base colour with per-vertex jitter; `lift` brightens with height
// (canopy tops catch more light).
function _paint(g, base, jitter, lift) {
  const p = g.attributes.position;
  const n = p.count;
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    const j = 1 + jitter * (2 * _hash3(x, y, z, 7) - 1);
    const l = 1 + (lift || 0) * clamp((y - 1) / 4.5, 0, 1);
    col[i * 3] = base[0] * j * l;
    col[i * 3 + 1] = base[1] * j * l;
    col[i * 3 + 2] = base[2] * j * l;
  }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return g;
}

// Tapered bark cylinder, origin at ground.
function _trunkGeo(h, rTop, rBot, seed) {
  const t = new THREE.CylinderGeometry(rTop, rBot, h, 7, 1).toNonIndexed();
  _displace(t, seed, 0.07);
  t.translate(0, h / 2, 0);
  _paint(t, [0.30, 0.21, 0.13], 0.14, 0.06);
  return t;
}

// A short branch stub angled out from the trunk at height `oy`.
function _branchGeo(oy, yaw, tilt, len, r, seed) {
  const b = new THREE.CylinderGeometry(r * 0.55, r, len, 5, 1).toNonIndexed();
  b.translate(0, len / 2, 0);
  b.rotateZ(tilt);
  b.rotateY(yaw);
  b.translate(0, oy, 0);
  _paint(b, [0.28, 0.19, 0.12], 0.12);
  return b;
}

// One displaced canopy blob.
function _blob(r, ox, oy, oz, sx, sy, sz, seed, base, amp) {
  const b = new THREE.IcosahedronGeometry(r, 1);
  _displace(b, seed, amp);
  b.scale(sx, sy, sz);
  b.translate(ox, oy, oz);
  _paint(b, base, 0.10, 0.09);
  return b;
}

// ---- tree species (metres, origin at ground level) --------------------------
// Broadleaf A — round, full, ~6 m tall: tapered trunk + 2 branches + 5 blobs.
function _broadleafGeoA(seed) {
  const B = [0.20, 0.38, 0.10];
  const parts = [_trunkGeo(3.1, 0.13, 0.24, seed)];
  parts.push(_branchGeo(2.6, 0.7, 0.9, 1.1, 0.09, seed + 31));
  parts.push(_branchGeo(2.8, -2.2, -1.0, 1.2, 0.08, seed + 47));
  parts.push(_blob(1.50, 0.00, 4.10, 0.00, 1.15, 0.90, 1.15, seed + 1, B, 0.20));
  parts.push(_blob(1.15, 1.05, 3.45, 0.35, 1.00, 0.85, 1.00, seed + 2, B, 0.22));
  parts.push(_blob(1.10, -0.95, 3.60, -0.45, 1.00, 0.85, 1.00, seed + 3, B, 0.22));
  parts.push(_blob(0.95, 0.25, 5.15, -0.40, 1.00, 0.80, 1.00, seed + 4, B, 0.20));
  parts.push(_blob(0.80, -0.35, 4.40, 0.95, 1.00, 0.75, 1.00, seed + 5, B, 0.22));
  return mergeGeometries(parts);
}

// Broadleaf B — wide, low "spread" form, ~4.5 m tall: short trunk + 5 flat blobs.
function _broadleafGeoB(seed) {
  const B = [0.23, 0.39, 0.11];
  const parts = [_trunkGeo(2.2, 0.12, 0.23, seed)];
  parts.push(_branchGeo(1.7, 1.9, 1.1, 1.0, 0.08, seed + 13));
  parts.push(_blob(1.50, 0.00, 2.85, 0.00, 1.28, 0.72, 1.28, seed + 1, B, 0.20));
  parts.push(_blob(1.05, 1.40, 2.50, 0.30, 1.00, 0.72, 1.00, seed + 2, B, 0.22));
  parts.push(_blob(1.00, -1.35, 2.60, -0.25, 1.00, 0.70, 1.00, seed + 3, B, 0.22));
  parts.push(_blob(0.90, 0.25, 3.55, -0.90, 1.00, 0.68, 1.00, seed + 4, B, 0.20));
  parts.push(_blob(0.72, -0.55, 3.30, 0.90, 1.00, 0.65, 1.00, seed + 5, B, 0.22));
  return mergeGeometries(parts);
}

// Conifer — 4 stacked, flattened, tapering cones to a point (~6.3 m).
function _coniferGeo(seed) {
  const parts = [_trunkGeo(1.7, 0.08, 0.17, seed)];
  const cones = [
    [2.00, 2.5, 2.35],
    [1.55, 2.2, 3.50],
    [1.05, 1.9, 4.60],
    [0.55, 1.55, 5.65],
  ];
  for (let i = 0; i < cones.length; i++) {
    const [r, h, oy] = cones[i];
    const c = new THREE.ConeGeometry(r, h, 8, 1).toNonIndexed();
    _displace(c, seed + i * 57, 0.13);
    c.scale(1, 0.82, 1); // flatten: branchy tiers, not party hats
    c.translate(0, oy, 0);
    const lift = 0.86 + i * 0.14;
    _paint(c, [0.115 * lift, 0.275 * lift, 0.13 * lift], 0.09, 0.06);
    parts.push(c);
  }
  return mergeGeometries(parts);
}

// Umbrella — tall slender trunk, high branching, one wide flat canopy crown
// (~5.9 m): the mature street tree (rain-tree / old plane look).
function _umbrellaGeo(seed) {
  const B = [0.22, 0.37, 0.10];
  const parts = [_trunkGeo(4.3, 0.11, 0.30, seed)];
  parts.push(_branchGeo(3.1, 0.5, 0.8, 1.3, 0.08, seed + 11));
  parts.push(_branchGeo(3.4, 2.6, -0.85, 1.3, 0.07, seed + 23));
  parts.push(_branchGeo(3.7, -1.7, 0.95, 1.2, 0.07, seed + 37));
  parts.push(_blob(1.35, 1.55, 4.70, 0.15, 1.05, 0.55, 1.05, seed + 1, B, 0.20));
  parts.push(_blob(1.30, -1.50, 4.65, -0.20, 1.05, 0.55, 1.05, seed + 2, B, 0.22));
  parts.push(_blob(1.30, 0.25, 4.80, 1.55, 1.05, 0.55, 1.05, seed + 3, B, 0.20));
  parts.push(_blob(1.25, -0.35, 4.75, -1.50, 1.05, 0.55, 1.05, seed + 4, B, 0.22));
  parts.push(_blob(1.45, 0.00, 5.05, 0.00, 1.10, 0.62, 1.10, seed + 5, B, 0.20));
  return mergeGeometries(parts);
}

// Streetlight: base + pole + arm + head housing, arm along local +X.
const POLE_HEAD = { x: 1.22, y: 6.78, z: 0 }; // local centre of the glow lens
function _poleGeo() {
  const T = THREE;
  const base = new T.CylinderGeometry(0.17, 0.21, 0.30, 8, 1).toNonIndexed();
  base.translate(0, 0.15, 0);
  _paint(base, [0.15, 0.16, 0.18], 0.05);
  const pole = new T.CylinderGeometry(0.065, 0.09, 6.6, 8, 1).toNonIndexed();
  pole.translate(0, 3.60, 0);
  _paint(pole, [0.20, 0.22, 0.25], 0.05);
  const collar = new T.CylinderGeometry(0.10, 0.075, 0.22, 8, 1).toNonIndexed();
  collar.translate(0, 6.96, 0);
  _paint(collar, [0.20, 0.22, 0.25], 0.05);
  const arm = new T.BoxGeometry(1.28, 0.085, 0.085).toNonIndexed();
  arm.rotateZ(-0.05);
  arm.translate(0.64, 6.95, 0);
  _paint(arm, [0.20, 0.22, 0.25], 0.05);
  const house = new T.BoxGeometry(0.56, 0.13, 0.32).toNonIndexed();
  house.rotateZ(-0.16);
  house.translate(POLE_HEAD.x, POLE_HEAD.y + 0.06, 0);
  _paint(house, [0.30, 0.32, 0.36], 0.04);
  return mergeGeometries([base, pole, collar, arm, house]);
}

// Glow lens: a small ellipsoid just under the head housing (the visible lamp).
function _headGeo() {
  const g = new THREE.IcosahedronGeometry(0.17, 1);
  g.scale(1.35, 0.5, 0.85);
  g.translate(POLE_HEAD.x, POLE_HEAD.y, 0);
  return g;
}

// Light pool: a flat quad laid on the asphalt, baked at the lamp-head offset so
// the instance transform matches the pole's. Radial falloff comes from the
// texture; the material colour ramps 0 (day) -> warm (night).
function _poolGeo() {
  const g = new THREE.PlaneGeometry(7.8, 7.8, 1, 1).toNonIndexed();
  g.rotateX(-Math.PI / 2);
  g.translate(POLE_HEAD.x, 0.07, 0); // just above ground: no z-fight
  return g;
}

// Radial gradient mask (white centre -> transparent edge) for the light pools.
function _poolTexture() {
  const S = 128;
  const cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const g = cv.getContext('2d');
  const grad = g.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
  grad.addColorStop(0.0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.22, 'rgba(255,255,255,0.8)');
  grad.addColorStop(0.5, 'rgba(255,255,255,0.32)');
  grad.addColorStop(0.78, 'rgba(255,255,255,0.08)');
  grad.addColorStop(1.0, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, S, S);
  return new THREE.CanvasTexture(cv);
}

// Bush: 3 low canopy blobs (~0.9 m tall). Stretched instances read as hedges.
function _bushGeo(seed) {
  const T = THREE;
  const blobs = [
    [0.55, 0.00, 0.42, 0.00, 1.30, 0.78, 1.30],
    [0.40, 0.42, 0.30, 0.28, 1.00, 0.72, 1.00],
    [0.38, -0.40, 0.28, -0.22, 1.00, 0.70, 1.00],
  ];
  const parts = [];
  for (let i = 0; i < blobs.length; i++) {
    const [r, ox, oy, oz, sx, sy, sz] = blobs[i];
    parts.push(_blob(r, ox, oy, oz, sx, sy, sz, seed + i * 83, [0.26, 0.40, 0.13], 0.16));
  }
  return mergeGeometries(parts);
}

// Bollard: short dark post with a pale reflective band.
function _bollardGeo() {
  const T = THREE;
  const body = new T.CylinderGeometry(0.09, 0.11, 0.90, 8, 1).toNonIndexed();
  body.translate(0, 0.45, 0);
  _paint(body, [0.16, 0.17, 0.19], 0.06);
  const band = new T.CylinderGeometry(0.095, 0.095, 0.15, 8, 1).toNonIndexed();
  band.translate(0, 0.60, 0);
  _paint(band, [0.72, 0.68, 0.55], 0.05);
  const cap = new T.CylinderGeometry(0.07, 0.088, 0.08, 8, 1).toNonIndexed();
  cap.translate(0, 0.94, 0);
  _paint(cap, [0.16, 0.17, 0.19], 0.06);
  return mergeGeometries([body, band, cap]);
}

// Bench: wood seat + slanted back on two dark steel legs; faces local +Z.
function _benchGeo() {
  const T = THREE;
  const seat = new T.BoxGeometry(1.9, 0.07, 0.52).toNonIndexed();
  seat.translate(0, 0.46, 0);
  _paint(seat, [0.44, 0.31, 0.19], 0.08);
  const back = new T.BoxGeometry(1.9, 0.36, 0.06).toNonIndexed();
  back.rotateX(-0.13);
  back.translate(0, 0.84, -0.24);
  _paint(back, [0.44, 0.31, 0.19], 0.08);
  const parts = [seat, back];
  for (const lx of [-0.8, 0.8]) {
    const leg = new T.BoxGeometry(0.09, 0.46, 0.5).toNonIndexed();
    leg.translate(lx, 0.23, 0);
    _paint(leg, [0.13, 0.14, 0.16], 0.05);
    parts.push(leg);
  }
  return mergeGeometries(parts);
}

export default class Props {
  name = 'props';

  constructor() {
    this._ctx = null; this._world = null; this._rng = null;
    this._root = null;
    this._geo = { tree: [], pole: null, head: null, pool: null, bush: null, bollard: null, bench: null };
    this._mat = { tree: null, pole: null, head: null, pool: null, bush: null, furn: null };
    this._tex = { pool: null };
    this._mesh = { tree: [], pole: null, head: null, pool: null, bush: null, bollard: null, bench: null };
    this._trees = []; this._lights = []; this._bushes = [];
    this._bollards = []; this._benches = [];
    this._treeCount = 0; this._lightCount = 0; this._bushCount = 0;
    this._bollardCount = 0; this._benchCount = 0;
    this._treeCap = 1400; this._lightCap = 420; this._bushCap = 380;
    this._bollardCap = 64; this._benchCap = 48;
    this._parkFbm = null;
    this._treeHash = new Map();
    this._bldHash = new Map();
    this._edges = [];
    // showcase staging
    this._showScene = null; this._showExtras = []; this._showSun = null; this._showHemi = null;
    this._terrainRoot = null; this._roadsRoot = null;
    this._bgDay = new THREE.Color(0x9db6d4);
    this._bgNight = new THREE.Color(0x070d18);
    // scratch (never allocated per frame)
    this._m = new THREE.Matrix4(); this._m2 = new THREE.Matrix4(); this._m3 = new THREE.Matrix4();
    this._q = new THREE.Quaternion();
    this._v = new THREE.Vector3(); this._vs = new THREE.Vector3(); this._w = new THREE.Vector3();
    this._col = new THREE.Color();
    this._yAxis = new THREE.Vector3(0, 1, 0);
  }

  async init(world, ctx) {
    this._ctx = ctx; this._world = world;
    this._rng = ctx.rng.fork('props');
    this._parkFbm = makeFbm(this._rng.fork('props-parks').seed);

    // The two NEW species draw their seeds from a side fork so the main
    // stream (broadleafA, conifer, bush — 3 draws, as originally) is
    // untouched and grove/light placement stays stable.
    const grng = this._rng.fork('props-geo');
    this._geo.tree = [
      _broadleafGeoA(this._rng.int(1, 100000)),
      _broadleafGeoB(grng.int(1, 100000)),
      _coniferGeo(this._rng.int(1, 100000)),
      _umbrellaGeo(grng.int(1, 100000)),
    ];
    this._geo.pole = _poleGeo();
    this._geo.head = _headGeo();
    this._geo.pool = _poolGeo();
    this._geo.bush = _bushGeo(this._rng.int(1, 100000));
    this._geo.bollard = _bollardGeo();
    this._geo.bench = _benchGeo();

    this._tex.pool = _poolTexture();
    this._mat.tree = new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.9, metalness: 0,
      emissive: 0x16300f, emissiveIntensity: 0, // night SSS-ish canopy bounce
    });
    this._mat.pole = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, metalness: 0.7 });
    this._mat.head = new THREE.MeshBasicMaterial({ color: 0x000000, toneMapped: false });
    this._mat.pool = new THREE.MeshBasicMaterial({
      map: this._tex.pool, color: 0x000000, transparent: true, opacity: 0.85,
      blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false,
    });
    this._mat.bush = new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.95, metalness: 0,
      emissive: 0x16300f, emissiveIntensity: 0,
    });
    this._mat.furn = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8, metalness: 0.15 });

    this._root = new THREE.Group();
    const mk = (geo, mat, cap, shadow) => {
      const m = new THREE.InstancedMesh(geo, mat, cap);
      m.count = 0;
      m.castShadow = shadow;
      m.frustumCulled = false; // instances span the whole map
      this._root.add(m);
      return m;
    };
    this._mesh.tree = [0, 1, 2, 3].map((i) =>
      mk(this._geo.tree[i], this._mat.tree, this._treeCap, true));
    this._mesh.pole = mk(this._geo.pole, this._mat.pole, this._lightCap, true);
    this._mesh.head = mk(this._geo.head, this._mat.head, this._lightCap, false);
    this._mesh.pool = mk(this._geo.pool, this._mat.pool, this._lightCap, false);
    this._mesh.pool.renderOrder = 2;
    this._mesh.bush = mk(this._geo.bush, this._mat.bush, this._bushCap, true);
    this._mesh.bollard = mk(this._geo.bollard, this._mat.furn, this._bollardCap, true);
    this._mesh.bench = mk(this._geo.bench, this._mat.furn, this._benchCap, true);

    const roads = ctx.registry && ctx.registry.get('roads');
    this._edges = (roads && roads.graph && roads.graph.edges) ? roads.graph.edges : [];
    this._buildBldHash();

    this._placeTrees();
    this._placeLights();
    this._placeStreetTrees();
    this._placeBushes();
    this._placeBollards();
    this._placeBenches();
    this._writeTrees();
    this._writeLights();
    this._writeBushes();
    this._writeFurniture();

    world.props = { trees: this._treeCount, lights: this._lightCount };
    ctx.scene.add(this._root);
    ctx.events.emit('props:ready', { trees: this._treeCount, lights: this._lightCount });
  }

  // ---- terrain / road / building queries ------------------------------------
  _hAt(x, z) {
    const t = this._world.terrain;
    const res = t.res, heights = t.heights, n = res + 1;
    const cell = t.size / res, half = t.size / 2;
    const gx = clamp((x + half) / cell, 0, res - 1e-4);
    const gz = clamp((z + half) / cell, 0, res - 1e-4);
    const x0 = gx | 0, z0 = gz | 0, fx = gx - x0, fz = gz - z0;
    const i00 = z0 * n + x0, i10 = i00 + 1, i01 = i00 + n, i11 = i01 + 1;
    const a = heights[i00] + (heights[i10] - heights[i00]) * fx;
    const b = heights[i01] + (heights[i11] - heights[i01]) * fx;
    return a + (b - a) * fz;
  }

  _roadDist(x, z) {
    let bd = Infinity, px = x, pz = z;
    for (const e of this._edges) {
      const pts = e.pts;
      for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i], b = pts[i + 1];
        const abx = b.x - a.x, abz = b.z - a.z;
        const l2 = abx * abx + abz * abz;
        if (l2 === 0) continue;
        let tt = ((x - a.x) * abx + (z - a.z) * abz) / l2;
        tt = tt < 0 ? 0 : tt > 1 ? 1 : tt;
        const qx = a.x + abx * tt, qz = a.z + abz * tt;
        const d2 = (qx - x) * (qx - x) + (qz - z) * (qz - z);
        if (d2 < bd) { bd = d2; px = qx; pz = qz; }
      }
    }
    return { d: Math.sqrt(bd), px, pz };
  }

  // Cumulative-length point sampler for a road edge (shared by lights/trees).
  _edgeSampler(pts) {
    const cum = [0];
    for (let i = 1; i < pts.length; i++)
      cum.push(cum[i - 1] + Math.hypot(pts[i].x - pts[i - 1].x, pts[i].z - pts[i - 1].z));
    const L = cum[cum.length - 1];
    const at = (s) => {
      let i = 0;
      while (i < cum.length - 2 && cum[i + 1] < s) i++;
      const seg = (cum[i + 1] - cum[i]) || 1;
      const f = clamp((s - cum[i]) / seg, 0, 1);
      const a = pts[i], b = pts[i + 1];
      const dx = b.x - a.x, dz = b.z - a.z;
      const len = Math.hypot(dx, dz) || 1;
      return { x: a.x + dx * f, z: a.z + dz * f, tx: dx / len, tz: dz / len };
    };
    return { L, at };
  }

  _buildBldHash() {
    const reg = this._ctx.registry;
    const bldsMod = reg && reg.get('buildings');
    const blds = Array.isArray(this._world.buildings) && this._world.buildings.length
      ? this._world.buildings
      : (bldsMod && Array.isArray(bldsMod._placed) ? bldsMod._placed : []);
    const m = this._bldHash;
    m.clear();
    for (const b of blds) {
      const hw = b.w / 2 + 2, hd = b.d / 2 + 2; // grow footprint by ~2 m
      const x0 = Math.floor((b.x - hw) / 8), x1 = Math.floor((b.x + hw) / 8);
      const z0 = Math.floor((b.z - hd) / 8), z1 = Math.floor((b.z + hd) / 8);
      for (let ix = x0; ix <= x1; ix++) for (let iz = z0; iz <= z1; iz++) {
        const k = ix + ':' + iz;
        let a = m.get(k);
        if (!a) { a = []; m.set(k, a); }
        a.push(b);
      }
    }
  }

  _inBuilding(x, z) {
    const ix0 = Math.floor((x - 4) / 8), ix1 = Math.floor((x + 4) / 8);
    const iz0 = Math.floor((z - 4) / 8), iz1 = Math.floor((z + 4) / 8);
    for (let ix = ix0; ix <= ix1; ix++) for (let iz = iz0; iz <= iz1; iz++) {
      const a = this._bldHash.get(ix + ':' + iz);
      if (!a) continue;
      for (let i = 0; i < a.length; i++) {
        const b = a[i];
        if (Math.abs(b.x - x) < b.w / 2 + 2.5 && Math.abs(b.z - z) < b.d / 2 + 2.5) return true;
      }
    }
    return false;
  }

  _treeNear(x, z, minD) {
    const cx = Math.floor(x / 7), cz = Math.floor(z / 7);
    for (let ix = cx - 1; ix <= cx + 1; ix++) for (let iz = cz - 1; iz <= cz + 1; iz++) {
      const a = this._treeHash.get(ix + ':' + iz);
      if (!a) continue;
      for (let i = 0; i < a.length; i++) {
        const t = a[i];
        const dx = t.x - x, dz = t.z - z;
        if (dx * dx + dz * dz < minD * minD) return true;
      }
    }
    return false;
  }

  _addTreeHash(t) {
    const k = Math.floor(t.x / 7) + ':' + Math.floor(t.z / 7);
    let a = this._treeHash.get(k);
    if (!a) { a = []; this._treeHash.set(k, a); }
    a.push(t);
  }

  // Full spot validation; returns ground height or null.
  _treeSpotOK(x, z) {
    const h = this._hAt(x, z);
    if (h < -1 || h > 40) return null; // no water, no rocky peaks
    // 10 m baseline slope guard: trees tolerate ~0.6 grade, not cliffs
    const hx = this._hAt(x + 5, z) - this._hAt(x - 5, z);
    const hz = this._hAt(x, z + 5) - this._hAt(x, z - 5);
    if (Math.hypot(hx, hz) > 6.0) return null;
    if (this._roadDist(x, z).d < 6) return null;
    if (this._inBuilding(x, z)) return null;
    if (this._treeNear(x, z, 2.6)) return null;
    return h;
  }

  _parkAt(x, z) { return this._parkFbm(x * 0.013 + 3.7, z * 0.013 - 1.9); }

  // Per-tree deterministic attributes from the site position. These draw
  // NOTHING from the RNG stream (only jitter + chance do), so grove density
  // stays stable across edits; shape/tint variation is a pure fn of (x, z).
  _treeAttrs(x, z) {
    const ix = Math.round(x * 127.1) | 0, iz = Math.round(z * 311.7) | 0;
    const h = (i) => {
      let s = (Math.imul(ix, 374761393) ^ Math.imul(iz, 668265263) ^ Math.imul(i + 1, 974634799)) >>> 0;
      s = (s ^ (s >>> 13)) >>> 0; s = Math.imul(s, 0x5bd1e995) >>> 0; s ^= s >>> 15;
      return (s >>> 0) / 4294967296;
    };
    return [h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8), h(9)];
  }

  // ---- placement --------------------------------------------------------------
  // Coarse 7 m gate cells (height/slope/building — slow-varying) + 4 jittered
  // 3.5 m sub-candidates per passing cell, each re-checked against the road
  // buffer and the spacing hash. Park pockets (low-freq fBm) get near-forest
  // density so groves read as groves; the rest of the green gets lawn scatter.
  _placeTrees() {
    const rng = this._rng;
    const list = this._trees;
    const step = 7, sub = 3.5;
    const add = (x, z, h, park, scaleMul) => {
      if (list.length >= this._treeCap) return;
      const r = Math.hypot(x, z);
      const hill = r > 150 || h > 14;
      const coniferP = park ? 0.16 : (hill ? 0.72 : 0.30);
      const umbrellaP = park ? 0.10 : 0.05;
      // Shape/tint from the position hash (stream-independent): a given spot
      // always grows the same tree, whatever the placement history.
      const A = this._treeAttrs(x, z);
      const variant = A[0] < coniferP ? 2 : A[0] < coniferP + umbrellaP ? 3 : (A[1] < 0.5 ? 0 : 1);
      const scale = (park ? 0.9 + A[2] * 0.9 : 0.6 + A[2] * 0.7) * scaleMul;
      // anisotropic silhouette jitter: wide/low vs narrow/tall (conifers stay
      // upright, umbrellas stay wide)
      let sx = 0.85 + A[3] * 0.35;
      let sy = 0.80 + A[4] * 0.45;
      let sz = 0.85 + A[5] * 0.35;
      if (variant === 2) { sx = 0.80 + A[3] * 0.30; sy = 0.85 + A[4] * 0.35; sz = 0.80 + A[5] * 0.30; }
      else if (variant === 3) { sx = 0.90 + A[3] * 0.28; sy = 0.90 + A[4] * 0.22; sz = 0.90 + A[5] * 0.28; }
      const t = {
        x, z, y: h, variant, scale, sx, sy, sz,
        rot: A[6] * Math.PI * 2,
        // green tint jitter: yellowish (low blue) to deep green (all low)
        cr: 0.78 + A[7] * 0.44,
        cg: 0.84 + A[8] * 0.34,
        cb: 0.60 + A[9] * 0.45,
      };
      list.push(t);
      this._addTreeHash(t);
    };

    for (let iz = -28; iz <= 28; iz++) {
      for (let ix = -28; ix <= 28; ix++) {
        const cx = ix * step, cz = iz * step;
        if (Math.hypot(cx, cz) > 195) continue;
        // coarse gates (slow-varying fields)
        const hc = this._hAt(cx, cz);
        if (hc < -1 || hc > 40) continue;
        const hx = this._hAt(cx + 5, cz) - this._hAt(cx - 5, cz);
        const hz = this._hAt(cx, cz + 5) - this._hAt(cx, cz - 5);
        if (Math.hypot(hx, hz) > 6.0) continue; // slope > ~0.6
        if (this._inBuilding(cx, cz)) continue;
        const park = this._parkAt(cx, cz) > 0.55;
        // sub-candidates: 2x2 scatter, or 4x4 forest density in park pockets
        const n4 = park ? 4 : 2;
        const csub = park ? 1.75 : sub;
        const jamp = park ? 1.4 : 2.4;
        for (let s = 0; s < n4 * n4; s++) {
          if (list.length >= this._treeCap) break;
          const x = cx + (s % n4) * csub + (rng.next() - 0.5) * jamp;
          const z = cz + ((s / n4) | 0) * csub + (rng.next() - 0.5) * jamp;
          const r = Math.hypot(x, z);
          if (r > 192) continue;
          const h = this._hAt(x, z);
          if (h < -1) continue;
          if (this._roadDist(x, z).d < 6) continue; // off road centerlines
          if (this._treeNear(x, z, 2.6)) continue;
          const p = park ? 0.92 : (r < 150 ? 0.95 : 0.15);
          if (!rng.chance(p)) continue;
          add(x, z, h, park, 1);
        }
      }
    }

    // Hero trees at the centre-block corners: guarantee a good massing for the
    // street / close-up camera sightlines (deterministic positions, mixed species).
    const heroes = [
      [20, 20, 0, 1.45], [-20, 20, 2, 1.25], [20, -20, 3, 1.30],
      [-20, -20, 1, 1.20], [20, 30, 3, 1.15], [-20, 30, 0, 1.25],
    ];
    for (let i = 0; i < heroes.length; i++) {
      if (list.length >= this._treeCap) break;
      const [hx, hz, varr, sc] = heroes[i];
      const h = this._treeSpotOK(hx, hz);
      if (h == null) continue;
      list.push({
        x: hx, z: hz, y: h, variant: varr, scale: sc,
        sx: 1.05, sy: 1.0, sz: 1.05,
        rot: rng.next() * Math.PI * 2,
        cr: 0.90 + rng.next() * 0.25, cg: 0.95 + rng.next() * 0.20, cb: 0.78 + rng.next() * 0.25,
      });
      this._addTreeHash(list[list.length - 1]);
    }
    this._treeCount = list.length;
  }

  _placeLights() {
    const rng = this._rng;
    const list = this._lights;
    const SPACING = 21; // metres along the edge
    for (let ei = 0; ei < this._edges.length; ei++) {
      const pts = this._edges[ei].pts;
      if (!pts || pts.length < 2) continue;
      const { L, at } = this._edgeSampler(pts);
      if (L < SPACING) continue;
      const phase = rng.next() * SPACING;
      const side0 = rng.chance(0.5) ? 1 : -1;
      let li = 0;
      for (let s = phase; s < L - 1; s += SPACING, li++) {
        if (list.length >= this._lightCap) break;
        const P = at(s);
        const nx = -P.tz, nz = P.tx; // left normal of travel
        const side = side0 * (li % 2 === 0 ? 1 : -1); // alternate sides
        const x = P.x + nx * 5.5 * side; // just outside the 10 m carriageway
        const z = P.z + nz * 5.5 * side;
        const h = this._hAt(x, z);
        if (h < -1) continue;
        const adx = -nx * side, adz = -nz * side; // arm points back at the road
        list.push({ x, y: h, z, yaw: Math.atan2(-adz, adx), scale: 0.96 + rng.next() * 0.08 });
      }
    }
    this._lightCount = list.length;
  }

  // Aligned rows of umbrella trees along one side of each road — the classic
  // avenue rhythm. Skips junctions and any lamp on the same side.
  _placeStreetTrees() {
    const rng = this._rng;
    const list = this._trees;
    const SPACING = 15, OFF = 5.9;
    for (let ei = 0; ei < this._edges.length; ei++) {
      const pts = this._edges[ei].pts;
      if (!pts || pts.length < 2) continue;
      const { L, at } = this._edgeSampler(pts);
      if (L < SPACING * 1.5) continue;
      const phase = rng.next() * SPACING;
      const side0 = rng.chance(0.5) ? 1 : -1;
      for (let s = phase; s < L - 2; s += SPACING) {
        if (list.length >= this._treeCap) return;
        const P = at(s);
        const nx = -P.tz, nz = P.tx;
        const x = P.x + nx * OFF * side0;
        const z = P.z + nz * OFF * side0;
        if (Math.hypot(x, z) > 195) continue;
        const h = this._hAt(x, z);
        if (h < -1 || h > 40) continue;
        if (this._roadDist(x, z).d < 6.4) continue; // junction / crossing
        if (this._inBuilding(x, z)) continue;
        if (this._treeNear(x, z, 3.2)) continue;
        // don't let a canopy swallow a streetlight
        let nearLamp = false;
        for (let i = 0; i < this._lights.length; i++) {
          const lp = this._lights[i];
          const dx = lp.x - x, dz = lp.z - z;
          if (dx * dx + dz * dz < 6.5 * 6.5) { nearLamp = true; break; }
        }
        if (nearLamp) continue;
        list.push({
          x, z, y: h, variant: 3,
          scale: rng.range(0.72, 1.05),
          sx: 0.95 + rng.next() * 0.2, sy: 0.92 + rng.next() * 0.16, sz: 0.95 + rng.next() * 0.2,
          rot: Math.atan2(-P.tz, P.tx), // yawed along the road: reads as a row
          cr: 0.80 + rng.next() * 0.40, cg: 0.86 + rng.next() * 0.30, cb: 0.62 + rng.next() * 0.40,
        });
        this._addTreeHash(list[list.length - 1]);
      }
    }
    this._treeCount = list.length;
  }

  _placeBushes() {
    const rng = this._rng;
    const list = this._bushes;
    const step = 7;
    for (let iz = -24; iz <= 24; iz++) {
      for (let ix = -24; ix <= 24; ix++) {
        const cx = ix * step, cz = iz * step;
        if (Math.hypot(cx, cz) > 175) continue;
        const hc = this._hAt(cx, cz);
        if (hc < -1 || hc > 30) continue;
        if (this._inBuilding(cx, cz)) continue;
        if (this._parkAt(cx, cz) <= 0.55) continue; // park pockets only
        if (!rng.chance(0.7)) continue;
        const n = 1 + (rng.chance(0.5) ? 1 : 0);
        for (let k = 0; k < n; k++) {
          if (list.length >= this._bushCap) break;
          const bx = cx + (rng.next() - 0.5) * 5.4;
          const bz = cz + (rng.next() - 0.5) * 5.4;
          const h = this._hAt(bx, bz);
          if (h < -1) continue;
          if (this._roadDist(bx, bz).d < 5) continue;
          const hedge = rng.chance(0.35);
          list.push({
            x: bx, y: h, z: bz,
            rot: hedge ? (rng.chance(0.5) ? 0 : Math.PI / 2) + (rng.next() - 0.5) * 0.5
                       : rng.next() * Math.PI * 2,
            sx: hedge ? rng.range(1.9, 2.8) : rng.range(0.7, 1.5),
            sy: hedge ? rng.range(0.42, 0.60) : rng.range(0.7, 1.3),
            sz: hedge ? rng.range(0.9, 1.3) : rng.range(0.7, 1.5),
            cr: 0.85 + rng.next() * 0.30, cg: 0.90 + rng.next() * 0.22, cb: 0.75 + rng.next() * 0.25,
          });
        }
      }
    }
    this._bushCount = list.length;
  }

  // Bollards on the sidewalks of the core's arterials — just outside the
  // lamps, 6.5 m off the centerline, a handful per edge.
  _placeBollards() {
    const rng = this._rng;
    const list = this._bollards;
    const OFF = 6.5;
    for (let ei = 0; ei < this._edges.length; ei++) {
      const pts = this._edges[ei].pts;
      if (!pts || pts.length < 2) continue;
      const { L, at } = this._edgeSampler(pts);
      if (L < 40) continue;
      const mid = at(L / 2);
      if (Math.hypot(mid.x, mid.z) > 55) continue; // core arterials only
      const side0 = rng.chance(0.5) ? 1 : -1;
      for (const s of [12, 18, 24, 30]) {
        if (list.length >= this._bollardCap) return;
        if (s > L - 8) break;
        const P = at(s);
        const nx = -P.tz, nz = P.tx;
        const x = P.x + nx * OFF * side0;
        const z = P.z + nz * OFF * side0;
        const h = this._hAt(x, z);
        if (h < -1) continue;
        if (this._inBuilding(x, z)) continue;
        if (this._roadDist(x, z).d < 6.5) continue; // junction
        let nearLamp = false;
        for (let i = 0; i < this._lights.length; i++) {
          const lp = this._lights[i];
          const dx = lp.x - x, dz = lp.z - z;
          if (dx * dx + dz * dz < 3.2 * 3.2) { nearLamp = true; break; }
        }
        if (nearLamp) continue;
        list.push({ x, y: h, z, rot: Math.atan2(-P.tz, P.tx), scale: 1 });
      }
    }
    this._bollardCount = list.length;
  }

  // Park benches: in the green, 6-35 m from a road, facing it.
  _placeBenches() {
    const rng = this._rng;
    const list = this._benches;
    const step = 12;
    for (let iz = -18; iz <= 18; iz++) {
      for (let ix = -18; ix <= 18; ix++) {
        if (list.length >= this._benchCap) return;
        const cx = ix * step, cz = iz * step;
        if (Math.hypot(cx, cz) > 160) continue;
        if (this._parkAt(cx, cz) <= 0.58) continue;
        if (!rng.chance(0.5)) continue;
        const x = cx + (rng.next() - 0.5) * 6;
        const z = cz + (rng.next() - 0.5) * 6;
        const h = this._hAt(x, z);
        if (h < -1) continue;
        const rd = this._roadDist(x, z);
        if (!isFinite(rd.d) || rd.d < 6 || rd.d > 35) continue;
        if (this._inBuilding(x, z)) continue;
        if (this._treeNear(x, z, 2.2)) continue;
        list.push({
          x, y: h, z,
          rot: Math.atan2(rd.px - x, rd.pz - z), // local +Z toward the road
          scale: 1,
        });
      }
    }
    this._benchCount = list.length;
  }

  // ---- instance writers (called on placement / setCount / addProp) ----------
  _writeTrees() {
    const meshes = this._mesh.tree;
    const n = Math.min(this._treeCount, this._trees.length, this._treeCap);
    const used = [0, 0, 0, 0];
    for (let i = 0; i < n; i++) {
      const t = this._trees[i];
      const m = meshes[t.variant];
      const slot = used[t.variant]++;
      this._q.setFromAxisAngle(this._yAxis, t.rot);
      this._v.set(t.x, t.y - 0.12, t.z); // sink base slightly (no floaters)
      this._vs.set(t.scale * t.sx, t.scale * t.sy, t.scale * t.sz);
      this._m.compose(this._v, this._q, this._vs);
      m.setMatrixAt(slot, this._m);
      this._col.setRGB(t.cr, t.cg, t.cb);
      m.setColorAt(slot, this._col);
    }
    for (let i = 0; i < 4; i++) {
      meshes[i].count = used[i];
      meshes[i].instanceMatrix.needsUpdate = true;
      if (meshes[i].instanceColor) meshes[i].instanceColor.needsUpdate = true;
    }
  }

  _writeLights() {
    const mp = this._mesh.pole, mh = this._mesh.head, mo = this._mesh.pool;
    const n = Math.min(this._lightCount, this._lights.length, this._lightCap);
    for (let i = 0; i < n; i++) {
      const L = this._lights[i];
      this._q.setFromAxisAngle(this._yAxis, L.yaw);
      this._v.set(L.x, L.y - 0.10, L.z);
      this._vs.setScalar(L.scale);
      this._m.compose(this._v, this._q, this._vs);
      mp.setMatrixAt(i, this._m);
      // light pool: same transform at the lamp base; the geometry carries the
      // head offset, so it lands on the asphalt right under the lens.
      mo.setMatrixAt(i, this._m);
      // glow lens: same transform, local head offset scaled + rotated into world
      this._w.set(POLE_HEAD.x, POLE_HEAD.y, POLE_HEAD.z).multiplyScalar(L.scale).applyQuaternion(this._q).add(this._v);
      this._m3.compose(this._w, this._q, this._vs);
      mh.setMatrixAt(i, this._m3);
    }
    mp.count = n; mh.count = n; mo.count = n;
    mp.instanceMatrix.needsUpdate = true;
    mh.instanceMatrix.needsUpdate = true;
    mo.instanceMatrix.needsUpdate = true;
  }

  _writeBushes() {
    const mb = this._mesh.bush;
    const n = Math.min(this._bushCount, this._bushes.length, this._bushCap);
    for (let i = 0; i < n; i++) {
      const b = this._bushes[i];
      this._q.setFromAxisAngle(this._yAxis, b.rot);
      this._v.set(b.x, b.y - 0.05, b.z);
      this._vs.set(b.sx, b.sy, b.sz);
      this._m.compose(this._v, this._q, this._vs);
      mb.setMatrixAt(i, this._m);
      this._col.setRGB(b.cr, b.cg, b.cb);
      mb.setColorAt(i, this._col);
    }
    mb.count = n;
    mb.instanceMatrix.needsUpdate = true;
    if (mb.instanceColor) mb.instanceColor.needsUpdate = true;
  }

  _writeFurniture() {
    const mb = this._mesh.bollard, mn = this._mesh.bench;
    const nb = Math.min(this._bollardCount, this._bollards.length, this._bollardCap);
    for (let i = 0; i < nb; i++) {
      const b = this._bollards[i];
      this._q.setFromAxisAngle(this._yAxis, b.rot);
      this._v.set(b.x, b.y - 0.04, b.z);
      this._vs.setScalar(b.scale);
      this._m.compose(this._v, this._q, this._vs);
      mb.setMatrixAt(i, this._m);
    }
    mb.count = nb;
    mb.instanceMatrix.needsUpdate = true;
    const nn = Math.min(this._benchCount, this._benches.length, this._benchCap);
    for (let i = 0; i < nn; i++) {
      const b = this._benches[i];
      this._q.setFromAxisAngle(this._yAxis, b.rot);
      this._v.set(b.x, b.y - 0.04, b.z);
      this._vs.setScalar(b.scale);
      this._m.compose(this._v, this._q, this._vs);
      mn.setMatrixAt(i, this._m);
    }
    mn.count = nn;
    mn.instanceMatrix.needsUpdate = true;
  }

  // 0 by day -> 1 at night; ramps at dawn (0.25-0.35) and dusk (0.85-0.95).
  _lightFactor(t) {
    if (t < 0.25) return 1;
    if (t < 0.35) return 1 - smooth01((t - 0.25) * 10);
    if (t < 0.85) return 0;
    if (t < 0.95) return smooth01((t - 0.85) * 10);
    return 1;
  }

  update(dt, world) {
    if (!this._root || !this._ctx) return;
    // keep our meshes in whichever scene is current (live <-> showcase)
    const target = this._showScene || this._ctx.scene;
    if (this._root.parent !== target) target.add(this._root);
    if (this._showScene) {
      if (this._terrainRoot && this._terrainRoot.parent !== this._showScene) this._showScene.add(this._terrainRoot);
      if (this._roadsRoot && this._roadsRoot.parent !== this._showScene) this._showScene.add(this._roadsRoot);
      this._stageLighting();
    }
    // Night ramp: ONE material colour for the glow heads, ONE for the light
    // pools, two emissive intensities — the entire per-frame cost.
    // toneMapped:false feeds the bloom pass.
    const lf = this._lightFactor(this._ctx.clock.t);
    const k = 5.2 * lf;
    this._mat.head.color.setRGB(0.99 * k, 0.82 * k, 0.55 * k);
    this._mat.pool.color.setRGB(1.7 * lf, 1.28 * lf, 0.78 * lf);
    this._mat.tree.emissiveIntensity = 0.4 * lf;
    this._mat.bush.emissiveIntensity = 0.35 * lf;
  }

  // ---- public API -------------------------------------------------------------
  addProp(type, pos) {
    if (!this._root || !pos) return null;
    const rng = this._rng;
    if (type === 'tree') {
      if (this._trees.length >= this._treeCap) return null;
      const x = pos.x, z = pos.z;
      const h = this._hAt(x, z);
      const t = {
        x, z, y: h,
        variant: rng.chance(0.4) ? (rng.chance(0.5) ? 2 : 3) : (rng.chance(0.5) ? 0 : 1),
        scale: rng.range(0.7, 1.4),
        sx: 0.85 + rng.next() * 0.35, sy: 0.80 + rng.next() * 0.45, sz: 0.85 + rng.next() * 0.35,
        rot: rng.next() * Math.PI * 2,
        cr: 0.85 + rng.next() * 0.30, cg: 0.90 + rng.next() * 0.25, cb: 0.72 + rng.next() * 0.28,
      };
      this._trees.push(t);
      this._addTreeHash(t);
      this._treeCount = this._trees.length;
      this._writeTrees();
    } else if (type === 'light') {
      if (this._lights.length >= this._lightCap) return null;
      const x = pos.x, z = pos.z;
      const h = this._hAt(x, z);
      const rd = this._roadDist(x, z);
      let adx = 1, adz = 0;
      if (isFinite(rd.d) && rd.d > 0.01) {
        const l = Math.hypot(rd.px - x, rd.pz - z) || 1;
        adx = (rd.px - x) / l; adz = (rd.pz - z) / l; // arm toward the road
      }
      this._lights.push({ x, y: h, z, yaw: Math.atan2(-adz, adx), scale: 1 });
      this._lightCount = this._lights.length;
      this._writeLights();
    } else {
      return null;
    }
    this._world.props = { trees: this._treeCount, lights: this._lightCount };
    this._ctx.events.emit('props:added', { type, pos: { x: pos.x, z: pos.z } });
    return true;
  }

  setCount(type, n) {
    if (!this._root) return;
    n = Math.max(0, Math.floor(Number(n) || 0));
    if (type === 'tree') {
      this._treeCount = Math.min(n, this._trees.length, this._treeCap);
      this._writeTrees();
    } else if (type === 'light') {
      this._lightCount = Math.min(n, this._lights.length, this._lightCap);
      this._writeLights();
    } else if (type === 'bush') {
      this._bushCount = Math.min(n, this._bushes.length, this._bushCap);
      this._writeBushes();
    } else {
      return;
    }
    this._world.props = { trees: this._treeCount, lights: this._lightCount };
    this._ctx.events.emit('props:count', { type, n: type === 'tree' ? this._treeCount : this._lightCount });
  }

  stats() {
    return {
      drawCalls: 10,
      trees: this._treeCount, lights: this._lightCount, bushes: this._bushCount,
      bollards: this._bollardCount, benches: this._benchCount,
      notes: '10 InstancedMesh: 4 tree species (2 broadleaf massings, conifer, umbrella street rows) + pole, glow heads, additive night light pools, hedges, bollards, benches',
    };
  }

  // ---- showcase (gauntlet staging) ---------------------------------------------
  showcase(scene, world, ctx) {
    this._showScene = scene;
    // remove the harness's foreign staging meshes (flat grass ground)
    for (const c of [...scene.children]) {
      if (c.isMesh) scene.remove(c);
    }
    // MOVE the terrain + roads roots in (never clone); restored in dispose()
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

    // borrow the environment module's sky-derived IBL
    const env = reg && reg.get('environment');
    if (env && env._ibRT) { scene.environment = env._ibRT.texture; scene.environmentIntensity = 0.55; }

    // our own warm sun
    for (const o of this._showExtras) o.removeFromParent();
    this._showExtras.length = 0;
    this._showSun = null;
    this._showHemi = null;
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

    this._stageLighting();
  }

  // follow the clock in the showcase: sun -> 0, sky/fog -> near-black at night
  _stageLighting() {
    const s = this._showScene;
    if (!s) return;
    const lf = this._lightFactor(this._ctx.clock.t);
    const day = 1 - lf;
    if (this._showSun) this._showSun.intensity = 2.0 * day;
    if (this._showHemi) this._showHemi.intensity = 0.35 * (0.22 + 0.78 * day);
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
    for (const g of this._geo.tree) if (g) g.dispose();
    for (const k of ['pole', 'head', 'pool', 'bush', 'bollard', 'bench']) if (this._geo[k]) this._geo[k].dispose();
    if (this._tex.pool) this._tex.pool.dispose();
    for (const k of ['tree', 'pole', 'head', 'pool', 'bush', 'furn']) if (this._mat[k]) this._mat[k].dispose();
    this._root = null;
    this._geo = { tree: [], pole: null, head: null, pool: null, bush: null, bollard: null, bench: null };
    this._mat = { tree: null, pole: null, head: null, pool: null, bush: null, furn: null };
    this._tex = { pool: null };
    this._mesh = { tree: [], pole: null, head: null, pool: null, bush: null, bollard: null, bench: null };
    this._showScene = null;
  }
}
