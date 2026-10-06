// zoning — land-use + density grid + the classic CS2-style translucent overlay.
//
// A deterministic master plan (dense commercial core, residential rings,
// industrial waterfront/edge pockets, rural gaps) is painted onto an 8 m tile
// grid over the lowlands (±140 m, 35×35) and rendered as ONE merged,
// height-snapped, vertex-coloured quad overlay — 1 draw call, no programmer art.
//
// Buildable = above water, below the rock line, not cliff-steep, and clear of
// every road centerline by 7 m, so the plan reads as city blocks, not paint.
//
// Determinism: layout comes only from ctx.rng (fork 'zoning'). No Math.random.
// Public API: setZone(tile,use,density), getZone(tile), tiles, overlay,
// forEachZoned(cb). Emits 'zoning:changed' (once at init with full state, and
// on every set) and 'zoning:ready' {tiles}.
import * as THREE from 'three';

const TILES = 35;        // 35×35 grid
const HALF = 140;        // span ±140 m
const TSIZE = 8;         // tile size (m)
const QUAD = 7.6;        // overlay quad (m) — 0.4 m seams, reads as a crisp grid
const LIFT = 0.35;       // hover above the ground (m)
const ROAD_CLEAR = 7;    // keep-out from road centerlines (m)

const USES = ['residential', 'commercial', 'industrial'];
const USE_RGB = {
  residential: [86 / 255, 190 / 255, 110 / 255],  // green
  commercial: [70 / 255, 130 / 255, 230 / 255],   // blue
  industrial: [235 / 255, 150 / 255, 60 / 255],   // orange
};
const DEFAULT_DENSITY = { residential: 2, commercial: 4, industrial: 1 };
const _hsl = { h: 0, s: 0, l: 0 }; // colour scratch (no per-tile allocation)

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

// 2-D value noise on an integer lattice, driven by the (stateless) RNG.hash2 so
// the field is a pure function of the forked seed.
function makeVNoise(N) {
  const lat = (ix, iz) => N.hash2(ix, iz);
  return (x, y) => {
    const ix = Math.floor(x), iz = Math.floor(y);
    const fx = x - ix, fy = y - iz;
    const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
    const a = lat(ix, iz), b = lat(ix + 1, iz), c = lat(ix, iz + 1), d = lat(ix + 1, iz + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  };
}

// Squared point->segment distance in XZ.
function segDist2(px, pz, ax, az, bx, bz) {
  const vx = bx - ax, vz = bz - az;
  const len2 = vx * vx + vz * vz || 1e-6;
  const t = clamp(((px - ax) * vx + (pz - az) * vz) / len2, 0, 1);
  const dx = px - (ax + vx * t), dz = pz - (az + vz * t);
  return dx * dx + dz * dz;
}

export default class Zoning {
  name = 'zoning';

  constructor() {
    this.ctx = null; this.world = null;
    this.tiles = [];            // all 1225 tiles {tx,tz,x,z,use,density,buildable,hC}
    this.overlay = null;        // the ONE merged mesh
    this._mat = null;
    this._zonedCount = 0;
    this._borrowed = [];        // terrain/roads roots moved into the showcase
    this._showExtras = [];      // showcase sun + target
    this._showSun = null;
    this._showScene = null;
  }

  async init(world, ctx) {
    this.ctx = ctx; this.world = world;
    this._buildGrid();
    this._masterPlan();
    this._buildMesh();
    ctx.scene.add(this.overlay);
    this._syncWorld();
    ctx.events.emit('zoning:changed', { full: true, tiles: this._zonedCount, uses: this._useCounts() });
    ctx.events.emit('zoning:ready', { tiles: this._zonedCount });
  }

  update(dt, world) {
    // Daytime feature: nothing animates in the live scene. In the showcase the
    // borrowed sun tracks the clock so the overlay reads under moving light.
    if (!this._showSun || !this.ctx) return;
    const sd = this.ctx.clock.sunDir();
    const day = this.ctx.clock.daylight();
    this._showSun.position.set(sd.x * 350, Math.max(80, sd.y * 350), sd.z * 350);
    this._showSun.intensity = 0.35 + 1.65 * day;
  }

  // The height/normal queries live on the terrain MODULE (registry), while the
  // raw buffers (waterLevel, heights, res, size) live on world.terrain.
  _terr() { return this.ctx.registry && this.ctx.registry.get('terrain'); }

  // ---- grid + buildability ----------------------------------------------------
  _buildGrid() {
    const tn = this.world.terrain;
    const terr = this._terr();
    const roads = this.ctx.registry && this.ctx.registry.get('roads');
    const edges = roads ? roads.graph.edges : [];
    const wl = tn.waterLevel;
    const lim = ROAD_CLEAR * ROAD_CLEAR;
    this.tiles = [];
    for (let tz = 0; tz < TILES; tz++) {
      for (let tx = 0; tx < TILES; tx++) {
        const x = -HALF + tx * TSIZE + TSIZE / 2;
        const z = -HALF + tz * TSIZE + TSIZE / 2;
        const h = QUAD / 2;
        // 5 height samples: 4 quad corners + centre
        const sx = [x - h, x + h, x + h, x - h, x];
        const sz = [z - h, z - h, z + h, z + h, z];
        let hMin = Infinity, hC = 0;
        for (let i = 0; i < 5; i++) {
          const hh = terr.getHeight(sx[i], sz[i]);
          if (hh < hMin) hMin = hh;
          if (i === 4) hC = hh;
        }
        // full-resolution road centerline clearance (every segment of every edge)
        let onRoad = false;
        for (const e of edges) {
          const p = e.pts;
          for (let i = 0; i + 1 < p.length; i++) {
            if (segDist2(x, z, p[i].x, p[i].z, p[i + 1].x, p[i + 1].z) < lim) { onRoad = true; break; }
          }
          if (onRoad) break;
        }
        const slopeOK = terr.getNormal(x, z).y > 0.88; // ~28° max
        // Keep the overlay off the beach: a plate hovering 0.4 m over a shore
        // that is 0.6 m over the sea reads as a floating decal from street level.
        // Require clearly-land heights so the grid sits on solid ground.
        const buildable = !onRoad && hC > wl + 3 && hC < 40 && hMin > wl + 2.2 && slopeOK;
        this.tiles.push({ tx, tz, x, z, use: null, density: 0, buildable, hC });
      }
    }
  }

  // ---- deterministic master plan ----------------------------------------------
  _masterPlan() {
    const N = this.ctx.rng.fork('zoning');
    const vnoise = makeVNoise(N);
    // commercial core sits on the centroid of the buildable land
    let cx = 0, cz = 0, cn = 0;
    for (const t of this.tiles) if (t.buildable) { cx += t.x; cz += t.z; cn++; }
    if (cn > 0) { cx /= cn; cz /= cn; }
    const R_CORE = 44, R_RES = 112;
    for (const t of this.tiles) {
      if (!t.buildable) continue;
      const d = Math.hypot(t.x - cx, t.z - cz);
      // slow + mid value noise warp the radii so no boundary is a hard ring
      const nC = vnoise(t.x * 0.016 + 3.7, t.z * 0.016 - 8.2);
      const nR = vnoise(t.x * 0.030 - 21.4, t.z * 0.030 + 11.9);
      const dCore = d * (0.72 + 0.56 * nC);
      const dRes = d * (0.78 + 0.44 * nR);
      let use = null, density = 0;
      if (dCore < R_CORE) {
        // commercial core, densest at the heart
        use = 'commercial';
        density = dCore < R_CORE * 0.55 ? 5 : 4;
        if (N.chance(0.15)) density = 4;
      } else if (dRes < R_RES && !N.chance(0.10)) {
        // residential ring; 3 near the core, 2 outboard; 10% left rural
        use = 'residential';
        density = dRes < 66 + 22 * nR ? 3 : 2;
      } else if (!N.chance(0.28) && (t.hC < 7 ? d > 40 : d > R_RES * 0.72)) {
        // industrial pockets: waterfront (low ground) + outer edges
        use = 'industrial';
        density = t.hC < 6 ? 2 : 1;
        if (N.chance(0.2)) density = density === 2 ? 1 : 2;
      }
      t.use = use;
      t.density = density;
    }
  }

  // ---- the ONE merged overlay mesh ---------------------------------------------
  _buildMesh() {
    const T = this.ctx.three;
    const terr = this._terr();
    const pos = [], col = [], idx = [];
    const c = new T.Color();
    this._zonedCount = 0;
    const h = QUAD / 2;
    for (const t of this.tiles) {
      if (!t.use) continue;
      this._zonedCount++;
      const base = pos.length / 3;
      // 4 corners + centre, snapped to terrain + LIFT (fan through the centre
      // follows gentle slopes with zero z-fighting)
      const sx = [t.x - h, t.x + h, t.x + h, t.x - h, t.x];
      const sz = [t.z - h, t.z - h, t.z + h, t.z + h, t.z];
      for (let i = 0; i < 5; i++) pos.push(sx[i], terr.getHeight(sx[i], sz[i]) + LIFT, sz[i]);
      // sRGB land-use colour -> linear working space. A mild saturation punch
      // (hue unchanged) keeps the palette legible over bright green lowlands;
      // density then lifts lightness toward white (up to +25% at density 5).
      const rgb = USE_RGB[t.use];
      c.setRGB(rgb[0], rgb[1], rgb[2], T.SRGBColorSpace);
      c.getHSL(_hsl);
      c.setHSL(_hsl.h, Math.min(1, _hsl.s * 1.30), _hsl.l);
      const k = 1 - (t.density / 5) * 0.25;
      c.r = c.r * k + (1 - k); c.g = c.g * k + (1 - k); c.b = c.b * k + (1 - k);
      for (let i = 0; i < 5; i++) col.push(c.r, c.g, c.b);
      idx.push(
        base, base + 1, base + 4,
        base + 1, base + 2, base + 4,
        base + 2, base + 3, base + 4,
        base + 3, base, base + 4,
      );
    }
    const geo = new T.BufferGeometry();
    geo.setAttribute('position', new T.Float32BufferAttribute(pos, 3));
    geo.setAttribute('color', new T.Float32BufferAttribute(col, 3));
    geo.setIndex(idx);
    geo.computeBoundingSphere();
    if (!this.overlay) {
      this._mat = new T.MeshBasicMaterial({
        vertexColors: true,
        // 0.5 (spec target ~0.38): the bright gauntlet terrain washes the
        // colours out a stop, so the overlay needs extra presence to read as
        // a clean CS2-style land-use map at orbit distance.
        transparent: true, opacity: 0.5,
        depthWrite: false, side: T.DoubleSide,
      });
      this.overlay = new T.Mesh(geo, this._mat);
      this.overlay.renderOrder = 2; // paints over terrain + water
    } else {
      this.overlay.geometry.dispose();
      this.overlay.geometry = geo;
    }
  }

  // ---- public API ----------------------------------------------------------------
  // tile: {tx,tz} or world {x,z}. use: 'residential'|'commercial'|'industrial'|null.
  setZone(tile, use, density) {
    const t = this._resolveTile(tile);
    if (!t) return null;
    if (use !== null && !USES.includes(use)) return t;
    t.use = use === null ? null : use;
    t.density = use === null ? 0 : clamp(Math.round(density ?? DEFAULT_DENSITY[use]), 1, 5);
    this._buildMesh();
    this._syncWorld();
    this.ctx.events.emit('zoning:changed', { tile: { tx: t.tx, tz: t.tz, x: t.x, z: t.z }, use: t.use, density: t.density });
    return t;
  }
  getZone(tile) {
    const t = this._resolveTile(tile);
    if (!t || !t.use) return null;
    return { use: t.use, density: t.density, tx: t.tx, tz: t.tz };
  }
  forEachZoned(cb) {
    for (const t of this.tiles) if (t.use) cb(t);
  }
  _resolveTile(tile) {
    if (!tile) return null;
    let tx, tz;
    if (Number.isInteger(tile.tx) && Number.isInteger(tile.tz)) { tx = tile.tx; tz = tile.tz; }
    else if (typeof tile.x === 'number' && typeof tile.z === 'number') {
      tx = Math.floor((tile.x + HALF) / TSIZE);
      tz = Math.floor((tile.z + HALF) / TSIZE);
    } else return null;
    if (tx < 0 || tx >= TILES || tz < 0 || tz >= TILES) return null;
    return this.tiles[tz * TILES + tx];
  }
  _useCounts() {
    const o = { residential: 0, commercial: 0, industrial: 0 };
    for (const t of this.tiles) if (t.use) o[t.use]++;
    return o;
  }
  // mirror the live grid into the shared world (other modules read world.zoning)
  _syncWorld() {
    const g = this.world.zoning.grid;
    g.clear();
    for (const t of this.tiles) if (t.use) g.set(`${t.tx},${t.tz}`, t);
  }

  // ---- showcase --------------------------------------------------------------------
  showcase(scene, world, ctx) {
    if (!this.overlay) return; // init failed; nothing to stage
    const T = ctx.three;
    // 1. drop foreign staging meshes (the harness's flat ground)
    for (const c of [...scene.children]) if (c.isMesh && c !== this.overlay) scene.remove(c);
    // 2. MOVE the real terrain + roads under the overlay (never clone)
    this._borrowed.length = 0;
    const terrain = ctx.registry && ctx.registry.get('terrain');
    const roads = ctx.registry && ctx.registry.get('roads');
    if (terrain && terrain._root) { terrain._root.removeFromParent(); scene.add(terrain._root); this._borrowed.push(terrain._root); }
    if (roads && roads._root) { roads._root.removeFromParent(); scene.add(roads._root); this._borrowed.push(roads._root); }
    // 3. borrow sky IBL
    const env = ctx.registry && ctx.registry.get('environment');
    if (env && env._ibRT) { scene.environment = env._ibRT.texture; scene.environmentIntensity = 0.55; }
    // 4. our own sun (update() tracks the clock)
    for (const o of this._showExtras) { o.removeFromParent(); if (o.geometry) o.geometry.dispose(); }
    this._showExtras.length = 0;
    this._showSun = new T.DirectionalLight(0xfff6ec, 2.0);
    this._showSun.position.set(200, 300, 140);
    this._showSun.castShadow = true;
    this._showSun.shadow.mapSize.set(2048, 2048);
    const sc = this._showSun.shadow.camera;
    sc.left = -260; sc.right = 260; sc.top = 260; sc.bottom = -260; sc.near = 20; sc.far = 900;
    sc.updateProjectionMatrix();
    this._showSun.shadow.bias = -0.0004;
    this._showSun.shadow.normalBias = 0.9;
    scene.add(this._showSun, this._showSun.target);
    this._showExtras.push(this._showSun, this._showSun.target);
    // the overlay itself
    this.overlay.removeFromParent();
    scene.add(this.overlay);
    this._showScene = scene;
  }

  dispose() {
    // give borrowed roots back to the live scene
    for (const r of this._borrowed) { if (this.ctx && this.ctx.scene) this.ctx.scene.add(r); }
    this._borrowed.length = 0;
    if (this._showScene) { this._showScene.environment = null; this._showScene = null; }
    for (const o of this._showExtras) { o.removeFromParent(); if (o.geometry) o.geometry.dispose(); }
    this._showExtras.length = 0;
    this._showSun = null;
    if (this.overlay) {
      this.overlay.removeFromParent();
      this.overlay.geometry.dispose();
      this._mat.dispose();
      this.overlay = null; this._mat = null;
    }
    this.tiles = [];
    this._zonedCount = 0;
  }

  stats() {
    return {
      drawCalls: 1,
      tiles: this._zonedCount,
      notes: '1 merged height-snapped quad overlay (vertex-coloured, 0.5 alpha) over the 35×35 · 8 m grid',
    };
  }
}
