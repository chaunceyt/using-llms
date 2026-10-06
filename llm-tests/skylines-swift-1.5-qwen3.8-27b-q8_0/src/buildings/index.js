// buildings — the city's massing: downtown towers, mid-rise ring, suburban
// houses, waterfront industry.
//
// ~300+ buildings merged into ONE BufferGeometry per class (residential /
// commercial / industrial) => 3 draw calls for the whole city (plus up to 3
// more for addBuilding() extras). Each class atlas carries FOUR facade style
// strips (warm brick, pale stucco, ribbon glass, dark panel, ...) with
// distinct palettes, window shapes (standard / small / ribbon / curtain wall)
// and ground-floor treatments; each building picks one style and offsets its
// UVs inside the strip, so no two neighbours share a pattern. A per-building
// vertex tint adds a further tonal shift. Facades read as designed at street
// scale: mullion grid lines, string courses, pilaster bays and window-pair
// variation break the blank panels; ground floors are recessed shopfront /
// entry bands with plinths, canopies and corner pilasters; roofs carry
// parapet coping, shadow eaves, equipment boxes and (on the tallest towers)
// spires. One shared emissive atlas drives the per-window + storefront night
// glow for every style.
//
// Placement: jittered 9 m block grid over the buildable mountain ring (h in
// (waterLevel+1, 40]), pushed clear of road centrelines, occupancy-checked,
// zoned by radius from the map centre (towers in, houses out, industry toward
// the edges/waterfront). The terrain is a mountain sloping to the sea, so
// each building's base is a tilted quad that hugs the ground at every corner
// (walls stay vertical) — a hillside city. A live zoning module, when it has
// data, overrides the distance rule per tile.
//
// NOTE: the heightfield's only buildable land is a ~8,000 m² slope in the
// mid-ring; the flat-land area caps the city well below a 300-800 target. See
// the module report for the terrain core-change request.
//
// Showcase: a dedicated dense downtown — ~200 varied towers (res/com/ind,
// 9-112 m, supertall bell at the centre) on the harness grass ground, aimed
// at the `skyline` camera, with its own clock-driven sun/moon + sky. The real
// mountain terrain is not staged (it dwarfs the city from that camera).
//
// Determinism: every layout value comes from ctx.rng.fork('buildings')
// (showcase cluster from the 'showcase' fork).
// Public API: buildFromZoning(), addBuilding(b), setLit(on), world.buildings.
// Emits buildings:placed, buildings:ready. Draw calls: 3-6.

import * as THREE from 'three';
import { makeAlbedoAtlas, makeEmissiveAtlas, ATLAS } from './atlas.js';
import { MergedBuilder, emitBuilding } from './geometry.js';

const c01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smooth01 = (a, b, x) => { const t = c01((x - a) / (b - a)); return t * t * (3 - 2 * t); };

// Per-class PBR + massing definition. `styles` are the four facade material
// families drawn into the class atlas (index -> strip A/B/C/D, whose window
// shape is fixed: standard / small / ribbon / curtain wall). `styleW` are the
// pick weights (residential skews warm, commercial glassy, industrial grey).
// Each style: wall/glass palette (0-255), frame light|dark, mullion 0..3,
// surface flag (brick/panel/rib/grain), spandrel band, ground treatment,
// awning flag, and the roof vertex tint.
export const CLASSES = {
  residential: {
    rough: 0.85, metal: 0.0, emissive: 0xffe0b0,
    styleW: [0.34, 0.34, 0.20, 0.12],
    awnings: [[168, 74, 52], [214, 200, 172], [120, 118, 82], [146, 90, 58]],
    styles: [
      // A — warm terracotta brick
      { wall: [178, 110, 84], glass: [64, 74, 86], frame: 'light', mullion: 1, brick: true, ground: 'door', awning: true, roof: [0.44, 0.36, 0.30] },
      // B — pale stucco
      { wall: [216, 204, 182], glass: [72, 86, 98], frame: 'dark', mullion: 0, ground: 'door', awning: true, roof: [0.38, 0.36, 0.33] },
      // C — warm beige slab, ribbon windows + balcony spandrels
      { wall: [198, 172, 138], glass: [84, 102, 118], frame: 'light', mullion: 3, spandrel: true, ground: 'door', roof: [0.35, 0.32, 0.28] },
      // D — dark rust brick block
      { wall: [134, 78, 60], glass: [54, 62, 72], frame: 'light', mullion: 1, brick: true, ground: 'door', roof: [0.30, 0.26, 0.24] },
    ],
  },
  commercial: {
    rough: 0.42, metal: 0.35, emissive: 0xcfe4f8,
    styleW: [0.38, 0.22, 0.20, 0.20],
    awnings: [[48, 108, 118], [150, 52, 48], [196, 158, 64], [44, 58, 86]],
    styles: [
      // A — blue-grey glass block
      { wall: [108, 124, 144], glass: [88, 118, 150], frame: 'dark', mullion: 2, ground: 'shop', roof: [0.30, 0.33, 0.38] },
      // B — pale concrete, punched windows
      { wall: [202, 196, 186], glass: [92, 112, 132], frame: 'dark', mullion: 0, ground: 'shop', roof: [0.35, 0.34, 0.32] },
      // C — steel-blue ribbon glass
      { wall: [152, 160, 172], glass: [72, 102, 140], frame: 'dark', mullion: 3, spandrel: true, ground: 'shop', roof: [0.30, 0.32, 0.36] },
      // D — dark panel curtain wall, warm glass
      { wall: [74, 79, 88], glass: [104, 120, 138], frame: 'dark', mullion: 0, panel: true, ground: 'shop', roof: [0.26, 0.27, 0.30] },
    ],
  },
  industrial: {
    rough: 0.82, metal: 0.12, emissive: 0xe9d6ae,
    styleW: [0.34, 0.24, 0.24, 0.18],
    awnings: null,
    styles: [
      // A — grey concrete
      { wall: [148, 150, 154], glass: [62, 68, 76], frame: 'dark', mullion: 2, ground: 'garage', roof: [0.30, 0.31, 0.33] },
      // B — rusted metal rib
      { wall: [142, 96, 70], glass: [60, 66, 74], frame: 'dark', mullion: 0, rib: true, ground: 'garage', roof: [0.30, 0.27, 0.25] },
      // C — light grey slab
      { wall: [178, 176, 170], glass: [66, 72, 80], frame: 'dark', mullion: 0, ground: 'garage', roof: [0.33, 0.33, 0.32] },
      // D — dark panel
      { wall: [106, 110, 116], glass: [58, 64, 72], frame: 'dark', mullion: 1, panel: true, ground: 'garage', roof: [0.27, 0.28, 0.30] },
    ],
  },
};

// Deterministic weighted style pick for a building.
function pickStyle(cls, rng) {
  const w = cls.styleW;
  const q = rng.next();
  let acc = 0;
  for (let i = 0; i < w.length; i++) { acc += w[i]; if (q < acc) return i; }
  return w.length - 1;
}

// Jittered placement grid over the buildable lowlands. The heightfield is a
// mountain ring with a narrow coastal plain, so the grid is fine (7 m) and the
// slope guard generous — the geometry lifts each building's corners to follow
// the terrain.
const GRID_STEP = 9; // block spacing ≈ footprint + street gap
const GRID_HALF = 170;
const SLOPE_MAX = 1.4; // global guard is lenient — per-building dH cap does the real work
const GROUND_H = 4.8; // metres of storefront band at street level

class Occupancy {
  constructor(margin = 0.3) { this.m = new Map(); this.margin = margin; }
  _cells(x, z, w, d, m) {
    const x0 = Math.floor((x - w / 2 - m) / 4), x1 = Math.floor((x + w / 2 + m) / 4);
    const z0 = Math.floor((z - d / 2 - m) / 4), z1 = Math.floor((z + d / 2 + m) / 4);
    for (let ix = x0; ix <= x1; ix++) for (let iz = z0; iz <= z1; iz++) {
      const k = ix + ':' + iz;
      let arr = this.m.get(k);
      if (!arr) { arr = []; this.m.set(k, arr); }
      arr.push([x, z, w, d]);
    }
  }
  add(b) { this._cells(b.x, b.z, b.w, b.d, this.margin); }
  overlaps(x, z, w, d) {
    const m = this.margin;
    const x0 = Math.floor((x - w / 2 - m) / 4), x1 = Math.floor((x + w / 2 + m) / 4);
    const z0 = Math.floor((z - d / 2 - m) / 4), z1 = Math.floor((z + d / 2 + m) / 4);
    for (let ix = x0; ix <= x1; ix++) for (let iz = z0; iz <= z1; iz++) {
      const arr = this.m.get(ix + ':' + iz);
      if (!arr) continue;
      for (const [bx, bz, bw, bd] of arr) {
        if (Math.abs(bx - x) < (bw + w) / 2 + m && Math.abs(bz - z) < (bd + d) / 2 + m) return true;
      }
    }
    return false;
  }
}

export default class Buildings {
  name = 'buildings';
  constructor() {
    this._root = null;
    this._meshes = {}; this._mats = {}; this._tex = {};
    this._placed = []; this._extraList = { residential: [], commercial: [], industrial: [] };
    this._extraMesh = {};
    this._extraDirty = false;
    this._lit = null; this._glow = 0;
    this._sc = null; this._scSun = null; this._scBg = null;
    this._scRoot = null; this._scGeo = null;
    this._scMoon = null; this._scHemi = null;
    this._rng = null;
  }

  async init(world, ctx) {
    const T = ctx.three;
    this.ctx = ctx; this.world = world;
    this._rng = ctx.rng.fork('buildings');

    this._makeTextures();
    this._makeMaterials();
    this._place();
    this._buildGeometry();

    this._root = new T.Group();
    for (const kind of Object.keys(CLASSES)) {
      const m = new T.Mesh(this._geo[kind], this._mats[kind]);
      m.castShadow = true;
      m.receiveShadow = true;
      this._meshes[kind] = m;
      this._root.add(m);
    }
    ctx.scene.add(this._root);

    world.buildings = this._placed;
    ctx.events.emit('buildings:placed', { buildings: this._placed });
    ctx.events.emit('buildings:ready', { count: this._placed.length });
  }

  // ---- procedural textures + materials ---------------------------------------
  _makeTextures() {
    const T = this.ctx.three;
    for (const kind of Object.keys(CLASSES)) {
      this._tex[kind] = makeAlbedoAtlas(T, CLASSES[kind], this._rng.fork('atlas-' + kind));
    }
    this._texEmissive = makeEmissiveAtlas(T, this._rng.fork('atlas-emissive'));
  }
  _makeMaterials() {
    const T = this.ctx.three;
    for (const kind of Object.keys(CLASSES)) {
      const cls = CLASSES[kind];
      this._mats[kind] = new T.MeshStandardMaterial({
        map: this._tex[kind],
        emissiveMap: this._texEmissive,
        emissive: new T.Color(cls.emissive),
        emissiveIntensity: 0,
        roughness: cls.rough,
        metalness: cls.metal,
        vertexColors: true,
      });
    }
  }

  // ---- terrain queries (bilinear over the shared heightfield) -----------------
  _hAt(x, z) {
    const t = this.world.terrain;
    const res = t.res, heights = t.heights, n = res + 1;
    const cell = t.size / res, half = t.size / 2;
    const gx = c01((x + half) / (res * cell)) * (res - 1e-4);
    const gz = c01((z + half) / (res * cell)) * (res - 1e-4);
    const x0 = gx | 0, z0 = gz | 0, fx = gx - x0, fz = gz - z0;
    const i00 = z0 * n + x0, i10 = i00 + 1, i01 = i00 + n, i11 = i01 + 1;
    const a = heights[i00] + (heights[i10] - heights[i00]) * fx;
    const b = heights[i01] + (heights[i11] - heights[i01]) * fx;
    return a + (b - a) * fz;
  }

  // Distance to the nearest road polyline; pushes the point clear of the
  // centreline (buildings sit in blocks between roads, not on them).
  _roadClear(x, z) {
    const roads = this.ctx.registry && this.ctx.registry.get('roads');
    const e = roads && roads.nearestEdge ? roads.nearestEdge(x, z) : null;
    if (!e || !e.pts || e.pts.length < 2) return { x, z, d: Infinity };
    let bd = Infinity, px = x, pz = z;
    const pts = e.pts;
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      const abx = b.x - a.x, abz = b.z - a.z;
      const l2 = abx * abx + abz * abz || 1;
      let tt = ((x - a.x) * abx + (z - a.z) * abz) / l2;
      tt = tt < 0 ? 0 : tt > 1 ? 1 : tt;
      const qx = a.x + abx * tt, qz = a.z + abz * tt;
      const dd = (qx - x) * (qx - x) + (qz - z) * (qz - z);
      if (dd < bd) { bd = dd; px = qx; pz = qz; }
    }
    const d = Math.sqrt(bd);
    // road half-width is 5m; keep building footprints clear of the carriageway
    // with a small margin so they read as lining the street, not on it.
    if (d < 6.5) return { x, z, d, skip: true }; // too close to the centreline
    if (d < 8) { const k = (8 - d) / d; x += (x - px) * k; z += (z - pz) * k; }
    return { x, z, d };
  }

  // ---- placement --------------------------------------------------------------
  // The heightfield is a central mountain ringed by a narrow coastal plain, so
  // the city is a RING: downtown towers hug the inner (steep) arc, mid-rise
  // fills the middle, houses + industry take the flat outer coast. Zoning is
  // radial from the map centre (0,0) — the mountain — which is the city's core.
  _place() {
    const rng = this._rng;
    const zoning = this.ctx.registry && this.ctx.registry.get('zoning');
    const waterMin = this.world.terrain.waterLevel + 1;
    const occ = new Occupancy(0.1);
    const tiles = [];
    // The terrain is now a flat city basin: the core (r~0) is buildable, so the
    // ring starts near the centre and spans the plain + lower hills (r<150).
    const R_IN = 8, R_OUT = 150;
    const R_SPAN = R_OUT - R_IN;

    // Pass 1: jittered grid, keep buildable tiles (terrain only).
    for (let iz = -Math.floor(GRID_HALF / GRID_STEP); iz <= Math.floor(GRID_HALF / GRID_STEP); iz++) {
      for (let ix = -Math.floor(GRID_HALF / GRID_STEP); ix <= Math.floor(GRID_HALF / GRID_STEP); ix++) {
        const x = ix * GRID_STEP + (rng.next() - 0.5) * 1.0;
        const z = iz * GRID_STEP + (rng.next() - 0.5) * 1.0;
        const r = Math.hypot(x, z);
        if (r < R_IN || r > R_OUT) continue;
        const h = this._hAt(x, z);
        if (h <= waterMin || h > 40) continue;
        // slope guard: reject the steepest ground (the geometry shear-lifts corners)
        const hx = this._hAt(x + 4, z) - this._hAt(x - 4, z);
        const hz = this._hAt(x, z + 4) - this._hAt(x, z - 4);
        if (Math.hypot(hx, hz) / 8 > SLOPE_MAX) continue;
        tiles.push({ x, z, h, r, ix, iz });
      }
    }
    if (!tiles.length) return;

    // Pass 2: roads, zoning, occupancy.
    const tryPlace = (x, z, kind, h, w, d) => {
      const rc = this._roadClear(x, z);
      if (rc.skip) return null;
      x = rc.x; z = rc.z;
      // footprint corners must stay on buildable ground
      const hs = [
        this._hAt(x - w / 2, z - d / 2), this._hAt(x + w / 2, z - d / 2),
        this._hAt(x + w / 2, z + d / 2), this._hAt(x - w / 2, z + d / 2),
      ];
      for (const hh of hs) if (hh <= waterMin) return null;
      const dH = Math.max(...hs) - Math.min(...hs);
      if (dH > 13.0) return null; // degenerate guard; base follows the slope
      if (occ.overlaps(x, z, w, d)) return null;
      // no two neighbouring buildings at the same height
      for (let i = this._placed.length - 1; i >= 0 && i > this._placed.length - 40; i--) {
        const b = this._placed[i];
        if (Math.abs(b.x - x) < 20 && Math.abs(b.z - z) < 20 && Math.abs(b.h - h) < 1.6) {
          h += (rng.next() < 0.5 ? -1 : 1) * (2 + rng.next() * 4);
          break;
        }
      }
      h = Math.max(5, h);
      const b = this._makeBuilding(x, z, w, d, h, kind);
      if (!b) return null;
      occ.add(b);
      this._placed.push(b);
      return b;
    };

    for (const t of tiles) {
      const band = c01((t.r - R_IN) / R_SPAN); // 0 = inner arc .. 1 = outer coast
      let kind = null, h = 0;

      // live zoning module overrides the distance rule when it has data
      let zone = null;
      if (zoning && typeof zoning.getZone === 'function') {
        try { zone = zoning.getZone(Math.round(t.x / this.world.tileSize), Math.round(t.z / this.world.tileSize)); }
        catch { zone = null; }
      }
      if (zone && CLASSES[zone.use]) {
        kind = zone.use;
        const dens = c01((zone.density || 3) - 1) / 4; // 0..1
        if (kind === 'commercial') h = 10 + dens * 34;
        else if (kind === 'residential') h = 6 + dens * 13;
        else h = 9 + dens * 9;
      } else if (band < 0.42) {
        // downtown core on the inner arc: tall towers on a sparse sub-grid so
        // their big footprints don't starve their neighbours, mid-rise in between.
        // Heights skew mid with a few supertalls for a real skyline.
        const sparse = ((t.ix + 2 * t.iz) % 3 === 0);
        if (sparse && rng.chance(0.9)) {
          kind = 'commercial';
          const superTall = rng.chance(0.28);
          h = superTall ? 58 + rng.next() * 38 : 30 + rng.next() * 26;
        } else if (rng.chance(0.6)) {
          kind = 'commercial';
          h = 12 + rng.next() * 14;
        } else {
          kind = 'residential';
          h = 9 + rng.next() * 10;
        }
      } else if (band < 0.72) {
        // mid-rise ring
        const q = rng.next();
        if (q < 0.40) { kind = 'commercial'; h = 10 + rng.next() * 15; }
        else if (q < 0.88) { kind = 'residential'; h = 7 + rng.next() * 11; }
        else { kind = 'industrial'; h = 9 + rng.next() * 8; }
      } else {
        // suburbs: houses, industry toward the edges / waterfront
        const pInd = 0.12 + 0.38 * ((band - 0.72) / 0.28) + (t.h < 8 ? 0.18 : 0);
        const q = rng.next();
        if (q < pInd) { kind = 'industrial'; h = 9 + rng.next() * 8; }
        else if (q < pInd + 0.10) { kind = 'commercial'; h = 8 + rng.next() * 7; }
        else { kind = 'residential'; h = 6 + rng.next() * 6; }
      }

      let w, d;
      if (kind === 'commercial') {
        if (h >= 30) { w = rng.range(7.5, 9.5); d = rng.range(7.5, 9.5); }
        else { w = rng.range(6.5, 7.8); d = rng.range(6.5, 7.8); }
      } else if (kind === 'residential') {
        if (h >= 14) { w = rng.range(6.5, 7.8); d = rng.range(6.5, 7.8); }
        else { w = rng.range(6, 7.5); d = rng.range(6, 7.5); }
      } else { w = rng.range(7, 8.5); d = rng.range(7, 8.5); }

      const b = tryPlace(t.x, t.z, kind, h, w, d);
      // detached second house in a small residential lot
      if (b && b.kind === 'residential' && b.h < 15 && b.w <= 12 && b.d <= 12 && rng.chance(0.42)) {
        const ox = (rng.next() < 0.5 ? -1 : 1) * rng.range(4.6, 6.4);
        const oz = (rng.next() < 0.5 ? -1 : 1) * rng.range(4.6, 6.4);
        tryPlace(t.x + ox, t.z + oz, 'residential', rng.range(5.5, 9), rng.range(5.5, 8), rng.range(5.5, 8));
      }
    }
  }

  _makeBuilding(x, z, w, d, h, kind, ground) {
    const rng = this._rng;
    const cls = CLASSES[kind];
    const g = ground || ((px, pz) => this._hAt(px, pz));
    const y0 = g(x, z) + 0.15;
    const tiers = [];
    if (h >= 52 && rng.chance(0.8)) {
      const h1 = h * rng.range(0.46, 0.60);
      const h2 = h * rng.range(0.22, 0.30);
      const s1 = rng.range(0.66, 0.80), s2 = rng.range(0.60, 0.76);
      tiers.push({ w, d, y0: 0, y1: h1 });
      tiers.push({ w: w * s1, d: d * s1, y0: h1, y1: h1 + (h - h1 - h2) });
      tiers.push({ w: w * s2, d: d * s2, y0: h1 + (h - h1 - h2), y1: h });
    } else if (h >= 24 && rng.chance(0.75)) {
      const h1 = h * rng.range(0.52, 0.70);
      const s1 = rng.range(0.64, 0.82);
      tiers.push({ w, d, y0: 0, y1: h1 });
      tiers.push({ w: w * s1, d: d * s1, y0: h1, y1: h });
    } else {
      tiers.push({ w, d, y0: 0, y1: h });
    }
    // absolute heights
    for (const t of tiers) { t.y0 += y0; t.y1 += y0; }
    let roof = null;
    if (kind === 'residential' && h < 15 && rng.chance(0.85)) {
      roof = { type: rng.chance(0.6) ? 'pyramid' : 'gable', h: rng.range(2.2, 4.0) };
    }
    const units = [];
    const top = tiers[tiers.length - 1];
    if (kind === 'commercial' && h >= 16 && rng.chance(0.55)) {
      const uw = top.w * rng.range(0.30, 0.40), ud = top.d * rng.range(0.30, 0.40);
      units.push({
        u: true, // mechanical penthouse (rooftop obstacle for detail)
        x: x + (rng.next() < 0.5 ? -1 : 1) * top.w * 0.18,
        z: z + (rng.next() < 0.5 ? -1 : 1) * top.d * 0.15,
        w: uw, d: ud, y0: top.y1, y1: top.y1 + rng.range(2.2, 3.4),
      });
    } else if (kind === 'industrial' && rng.chance(0.6)) {
      const uw = top.w * rng.range(0.28, 0.38), ud = top.d * rng.range(0.24, 0.34);
      units.push({
        u: true,
        x: x + (rng.next() < 0.5 ? -1 : 1) * top.w * 0.20,
        z: z + (rng.next() < 0.5 ? -1 : 1) * top.d * 0.18,
        w: uw, d: ud, y0: top.y1, y1: top.y1 + rng.range(1.8, 2.8),
      });
    }
    for (const u of units) tiers.push(u);
    // facade style + roof tint (deterministic per building)
    const style = pickStyle(cls, rng);
    const b = { x, z, y0, w, d, h, kind, tiers, roof, groundH: GROUND_H, style, roofTint: cls.styles[style].roof };
    // Follow the terrain at EVERY corner (up and down) so the base is a tilted
    // quad hugging the ground and the walls stay vertical — a proper hillside
    // building, no floating on the downhill side.
    b._off = (px, pz) => g(px, pz) + 0.15 - y0;
    return b;
  }

  _buildGeometry() {
    const T = this.ctx.three;
    const builders = {};
    for (const kind of Object.keys(CLASSES)) builders[kind] = new MergedBuilder();
    for (let i = 0; i < this._placed.length; i++) {
      emitBuilding(this._placed[i], builders[this._placed[i].kind], CLASSES[this._placed[i].kind],
        this._rng.fork('bld' + i));
    }
    this._geo = {};
    for (const kind of Object.keys(CLASSES)) this._geo[kind] = builders[kind].finish(T);
  }

  // ---- public API --------------------------------------------------------------
  buildFromZoning() {
    if (this._placed.length) return; // already placed
    this._place();
    this._buildGeometry();
    for (const kind of Object.keys(CLASSES)) {
      if (this._geo[kind]) {
        const m = new this.ctx.three.Mesh(this._geo[kind], this._mats[kind]);
        m.castShadow = true; m.receiveShadow = true;
        this._meshes[kind] = m;
        this._root.add(m);
      }
    }
    this.world.buildings = this._placed;
    this.ctx.events.emit('buildings:placed', { buildings: this._placed });
    this.ctx.events.emit('buildings:ready', { count: this._placed.length });
  }

  addBuilding(b) {
    if (!b || !isFinite(b.x) || !isFinite(b.z)) return;
    const kind = CLASSES[b.kind] ? b.kind : 'residential';
    const w = Math.min(Math.max(b.w || 10, 4), 30);
    const d = Math.min(Math.max(b.d || 10, 4), 30);
    const h = Math.min(Math.max(b.h || 10, 4), 95);
    const placed = this._makeBuilding(b.x, b.z, w, d, h, kind);
    if (!placed) return;
    this._extraList[kind].push(placed);
    this._extraDirty = true;
    this.world.buildings.push({ x: placed.x, z: placed.z, w, d, h, kind });
    this.ctx.events.emit('buildings:placed', { b: { x: placed.x, z: placed.z, w, d, h, kind } });
  }

  // Force the night windows on/off; null = follow the clock.
  setLit(on) { this._lit = on; }

  // ---- per-frame ---------------------------------------------------------------
  update(dt, world) {
    if (this._extraDirty) this._rebuildExtras();
    // night-window glow: ~0 at noon, ~1.6 at night (smoothed to avoid popping)
    const dl = this.ctx.clock.daylight();
    let target;
    if (this._lit === true) target = 1.4;
    else if (this._lit === false) target = 0;
    else target = 1.6 * c01((1 - dl) * 3 - 1);
    this._glow += (target - this._glow) * Math.min(1, dt * 2.5);
    if (target === 0 && this._glow < 0.002) this._glow = 0;
    for (const kind of Object.keys(this._mats)) this._mats[kind].emissiveIntensity = this._glow;
    if (this._sc) this._stageSky();
  }

  _rebuildExtras() {
    this._extraDirty = false;
    const T = this.ctx.three;
    for (const kind of Object.keys(this._extraList)) {
      const list = this._extraList[kind];
      const old = this._extraMesh[kind];
      if (old) { old.removeFromParent(); old.geometry.dispose(); this._extraMesh[kind] = null; }
      if (!list.length) continue;
      const builder = new MergedBuilder();
      for (let i = 0; i < list.length; i++) {
        emitBuilding(list[i], builder, CLASSES[kind], this._rng.fork('extra:' + kind + i));
      }
      const m = new T.Mesh(builder.finish(T), this._mats[kind]);
      m.castShadow = true; m.receiveShadow = true;
      this._extraMesh[kind] = m;
      this._root.add(m);
    }
  }

  // ---- showcase ------------------------------------------------------------------
  // A dense, self-contained downtown: a wall of ~200 varied towers built on
  // the harness's grass ground, aimed at the `skyline` camera. The real
  // mountain terrain is NOT staged here — from the skyline camera it dwarfs
  // the city and reads as a dark ridge with a few far-off towers. The
  // cluster reuses the class materials (and their clock-driven window glow),
  // merged into 3 draw calls, plus a clock-driven sun/moon for the light.
  showcase(scene, world, ctx) {
    this._sc = scene;
    const T = ctx.three;
    const rng = this._rng.fork('showcase');

    // The skyline camera sits at (200, 42, 300) looking at (0, 32, 0); its
    // view XZ direction is ~(-0.555, -0.832), so the tower wall runs along
    // (0.832, -0.555) and its depth axis (0.555, 0.832) points at the camera.
    const WA = { x: 0.832, z: -0.555 };
    const WD = { x: 0.555, z: 0.832 };
    const builders = {};
    for (const kind of Object.keys(CLASSES)) builders[kind] = new MergedBuilder();
    const flat = () => 0;
    const occ = new Occupancy(0.2);
    let n = 0;
    for (let j = -4; j <= 5; j++) {
      for (let i = -19; i <= 19; i++) {
        if (rng.chance(0.14)) continue; // plaza gaps break the wall up
        const gw = i * 11 + rng.range(-1.5, 1.5);
        const gd = j * 12 + rng.range(-1.5, 1.5);
        const x = WA.x * gw + WD.x * gd;
        const z = WA.z * gw + WD.z * gd;
        // downtown bell: supertalls at the centre, low blocks at the flanks.
        // Depth ramp: rows near the camera run lower so the back rows'
        // supertalls tower over them — layered depth, not a flat curtain.
        const bell = 18 + 88 * Math.exp(-(i * i) / 225);
        let h = bell * rng.range(0.55, 1.12) * (1.0 - 0.35 * ((j + 4) / 9));
        const q = rng.next();
        let kind;
        if (h > 45) kind = q < 0.78 ? 'commercial' : 'residential';
        else if (h > 24) kind = q < 0.42 ? 'commercial' : q < 0.78 ? 'residential' : 'industrial';
        else kind = q < 0.38 ? 'residential' : q < 0.62 ? 'commercial' : 'industrial';
        if (kind === 'industrial') h = Math.min(h, 22);
        h = Math.min(112, Math.max(9, h));
        let w, d;
        if (kind === 'commercial') { w = rng.range(9, h > 50 ? 13.5 : 11); d = rng.range(9, h > 50 ? 13.5 : 11); }
        else if (kind === 'residential') { w = rng.range(8, 10.5); d = rng.range(8, 10.5); }
        else { w = rng.range(9, 12); d = rng.range(8, 11); }
        if (occ.overlaps(x, z, w, d)) continue;
        // Keep the `street` preset (pos 8,9,42 -> look 0,6,-60) clear: a
        // camera bubble plus the avenue it looks down the length of.
        {
          const hw = w / 2 + 2.5, hd = d / 2 + 2.5;
          const bubble = x + hw > -2 && x - hw < 20 && z + hd > 28 && z - hd < 54;
          const avenue = x + hw > -4 && x - hw < 14 && z + hd > -34 && z - hd < 28;
          if (bubble || avenue) continue;
        }
        const b = this._makeBuilding(x, z, w, d, h, kind, flat);
        if (!b) continue;
        occ.add(b);
        emitBuilding(b, builders[kind], CLASSES[kind], rng.fork('sc' + n));
        n++;
      }
    }
    this._scGeo = {};
    this._scRoot = new T.Group();
    for (const kind of Object.keys(CLASSES)) {
      const geo = builders[kind].finish(T);
      this._scGeo[kind] = geo;
      const m = new T.Mesh(geo, this._mats[kind]);
      m.castShadow = true; m.receiveShadow = true;
      this._scRoot.add(m);
    }
    scene.add(this._scRoot);

    // Light: keep the harness hemisphere (dimmed at night), add a clock-driven
    // sun (shadows) and a faint moon fill so facades read after sunset.
    this._scHemi = null;
    for (const c of scene.children) if (c.isHemisphereLight) { this._scHemi = c; break; }
    const sun = new T.DirectionalLight(0xfff6ec, 2.0);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = -280; sc.right = 280; sc.top = 280; sc.bottom = -280; sc.near = 20; sc.far = 900;
    sc.updateProjectionMatrix();
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.9;
    scene.add(sun, sun.target);
    this._scSun = sun;
    const moon = new T.DirectionalLight(0x93a7cc, 0.14);
    moon.position.set(-140, 190, -90);
    scene.add(moon, moon.target);
    this._scMoon = moon;
    this._scBg = scene.background; // harness solid color; _stageSky drives it
    this._stageSky();
  }

  // Drive the showcase sun/moon + sky colours from the clock (day -> dusk -> night).
  _stageSky() {
    const T = this.ctx.three;
    const sc = this._sc;
    const sun = this._scSun;
    const clock = this.ctx.clock;
    const t = clock.t;
    const sd = clock.sunDir();
    const a = (t - 0.25) * Math.PI * 2;
    if (sun) {
      sun.position.set(Math.cos(a) * 240, Math.sin(a) * 240 + 26, 130);
      const up = c01((sd.elev + 0.06) / 0.45);
      sun.intensity = 2.1 * up;
      sun.visible = up > 0.002;
      if (sun.color) sun.color.setHex(0xfff6ec).lerp(new T.Color(0xffb066), 1 - c01(sd.elev * 2.6));
    }
    if (this._scMoon) this._scMoon.intensity = 0.16 * (1 - clock.daylight());
    if (this._scHemi) this._scHemi.intensity = 0.18 + 0.5 * clock.daylight();
    // sky: night -> day, with an orange band around sunrise/sunset
    if (this._scBg) {
      const day = new T.Color(0x9db6d4);
      const night = new T.Color(0x070d18);
      const dusk = new T.Color(0xef9c58);
      this._scBg.copy(night).lerp(day, smooth01(0, 1, clock.daylight()));
      const e = sd.elev;
      if (e > -0.07 && e < 0.32) this._scBg.lerp(dusk, Math.exp(-Math.pow(e / 0.12, 2)) * 0.8);
      if (sc.fog) sc.fog.color.copy(this._scBg);
    }
  }

  dispose() {
    const T = this.ctx.three;
    void T;
    // tear down the showcase cluster
    if (this._scRoot) {
      this._scRoot.removeFromParent();
      for (const kind of Object.keys(this._scGeo || {})) {
        if (this._scGeo[kind]) this._scGeo[kind].dispose();
      }
      this._scRoot = null; this._scGeo = null;
    }
    if (this._scSun) {
      this._scSun.removeFromParent();
      this._scSun.target.removeFromParent();
      if (this._scSun.shadow && this._scSun.shadow.map) this._scSun.shadow.map.dispose();
      this._scSun = null;
    }
    if (this._scMoon) { this._scMoon.removeFromParent(); this._scMoon.target.removeFromParent(); this._scMoon = null; }
    this._scHemi = null;
    if (this._root) { this._root.removeFromParent(); this._root = null; }
    for (const kind of Object.keys(CLASSES)) {
      if (this._geo && this._geo[kind]) this._geo[kind].dispose();
      if (this._mats[kind]) this._mats[kind].dispose();
      if (this._tex[kind]) this._tex[kind].dispose();
    }
    for (const kind of Object.keys(this._extraMesh)) {
      if (this._extraMesh[kind]) { this._extraMesh[kind].geometry.dispose(); this._extraMesh[kind] = null; }
    }
    if (this._texEmissive) this._texEmissive.dispose();
    this._sc = null;
  }

  stats() {
    let extras = 0;
    for (const kind of Object.keys(this._extraMesh)) if (this._extraMesh[kind]) extras++;
    return {
      drawCalls: 3 + extras,
      buildings: this.world.buildings ? this.world.buildings.length : 0,
      notes: `3 merged class meshes (res/com/ind) + ${extras} extras · 4-style facade strips per class atlas ${ATLAS.size}px · mullions/string courses/pilasters · recessed ground floors + plinths + canopies · parapet coping + eaves · roof units/spires/antennas/beacons/signage · night emissive`,
    };
  }
}
