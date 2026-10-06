// roads — the city's circulatory system.
//
// A deterministic network of lanes (a lowland grid + a few curved arterials)
// rendered as THREE merged ribbon geometries, all snapped to the terrain:
//   1. asphalt + lane markings (merged ribbon, procedural texture with tire-wear
//      channels, curb AO and pore mottling)
//   2. sidewalk/curb band (slightly wider merged ribbon 0.05m below the asphalt,
//      light concrete with a dark curb line baked at the asphalt edge)
//   3. zebra crosswalks (merged quads, one per approach at every grid intersection)
// Plus a lightweight graph (nodes/edges) that traffic, zoning and the sim query
// via samplePoint / nearestEdge.
//
// Determinism: layout comes only from ctx.rng (fork 'roads'). No Math.random.
// Performance: 3 draw calls in the live scene. Well under the 300 ceiling.
//
// Public API: addEdge(e), graph {nodes,edges}, samplePoint(edge,t), nearestEdge(x,z).
// Emits roads:edge (per road, {a:{x,z},b:{x,z}}) and roads:ready.
import * as THREE from 'three';

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

// deterministic hash in [0,1) — texture grain without Math.random
const hash = (n) => { const s = Math.sin(n) * 43758.5453; return s - Math.floor(s); };

// Build a flat ribbon (two triangles per segment) from a polyline in XZ, lifted
// to the given heights. Returns interleaved attribute arrays.
function ribbon(points, halfW, out) {
  const base = out.pos.length / 3;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const prev = points[Math.max(0, i - 1)];
    const next = points[Math.min(points.length - 1, i + 1)];
    let tx = next.x - prev.x, tz = next.z - prev.z;
    const tl = Math.hypot(tx, tz) || 1;
    tx /= tl; tz /= tl;
    const nx = -tz, nz = tx; // perpendicular in XZ
    out.pos.push(p.x + nx * halfW, p.y, p.z + nz * halfW, p.x - nx * halfW, p.y, p.z - nz * halfW);
    out.nrm.push(0, 1, 0, 0, 1, 0);
    const v = p._s; // arc-length in metres -> texture tiling along the road
    out.uv.push(0, v, 1, v);
    if (i < points.length - 1) {
      const a = base + i * 2, b = a + 1, c = a + 2, d = a + 3;
      out.idx.push(a, c, b, b, c, d);
    }
  }
}

export default class Roads {
  name = 'roads';
  constructor() {
    this._root = null; this._mesh = null; this._tex = null;
    this._sideMesh = null; this._sideTex = null;
    this._xwMesh = null; this._xwTex = null;
    this._showExtras = []; this._terrainRoot = null;
    this.graph = { nodes: [], edges: [] };
    this._intersections = [];
    this._hw = 5; // half road width (m) -> 10m wide, 2 lanes
    this._sw = 1.5; // sidewalk band beyond each road edge (m)
  }

  async init(world, ctx) {
    this.ctx = ctx; this.world = world;
    this._rng = ctx.rng.fork('roads');
    this._buildNetwork();
    this._build();
    ctx.scene.add(this._root);
    ctx.events.emit('roads:ready', { edges: this.graph.edges.length });
  }

  // Lay out the network: a lowland grid + two sweeping arterials.
  _buildNetwork() {
    const getH = (x, z) => this.ctx.world.terrain ? this._heightAt(x, z) : 0;
    const nodes = this.graph.nodes, edges = this.graph.edges;
    const road = (pts, kind) => {
      // resample to evenly spaced points and compute arc length for UV tiling
      const out = []; let s = 0;
      for (let i = 0; i < pts.length; i++) {
        if (i > 0) s += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].z - pts[i - 1].z);
        out.push({ x: pts[i].x, z: pts[i].z, y: getH(pts[i].x, pts[i].z) + 0.25, _s: s / 12 });
      }
      const a = pts[0], b = pts[pts.length - 1];
      const edge = { a: { x: a.x, z: a.z }, b: { x: b.x, z: b.z }, pts: out, kind };
      edges.push(edge);
      this.ctx.events.emit('roads:edge', { a: edge.a, b: edge.b });
      return edge;
    };

    const N = this._rng;
    // Span the buildable plain (terrain basin is ~r130m); 40m city blocks.
    const span = 240, step = 40, n = 7;
    // grid roads with a touch of deterministic jitter so it doesn't read robotic
    for (let i = 0; i < n; i++) {
      const o = -span / 2 + i * step + (N.next() - 0.5) * 6;
      const hz = [], vz = [];
      for (let j = 0; j <= 24; j++) {
        const t = -span / 2 + (j / 24) * span;
        const wobble = Math.sin(t * 0.02 + i) * 3;
        hz.push({ x: t, z: o + wobble });
        vz.push({ x: o + wobble, z: t });
      }
      road(hz, 'h'); road(vz, 'v');
    }
    // two curved arterials sweeping across the lowlands
    for (let k = 0; k < 2; k++) {
      const pts = [];
      const ph = N.next() * 6.28, amp = 40 + N.next() * 30;
      for (let j = 0; j <= 40; j++) {
        const t = (j / 40 - 0.5) * 2 * span;
        const y = Math.sin(t * 0.012 + ph) * amp;
        pts.push(k === 0 ? { x: t, z: y } : { x: y, z: t });
      }
      road(pts, 'a');
    }
    // dedup node list from edge endpoints (graph nodes)
    for (const e of edges) {
      if (!nodes.some((nd) => Math.abs(nd.x - e.a.x) < 1 && Math.abs(nd.z - e.a.z) < 1)) nodes.push({ ...e.a });
      if (!nodes.some((nd) => Math.abs(nd.x - e.b.x) < 1 && Math.abs(nd.z - e.b.z) < 1)) nodes.push({ ...e.b });
    }
    // where two grid streets cross, a crossing (for crosswalks)
    this._intersections = this._findIntersections();
  }

  // Find points where a horizontal and a vertical grid edge pass within 4.5m of
  // each other (the ±3m wobble can push a crossing pair ~4m apart at closest
  // approach, so 3m misses a few). Edges are resampled at 2m so the wobble is
  // resolved and the closest approach is accurate. Deterministic: fixed
  // iteration order only.
  _findIntersections() {
    const hs = this.graph.edges.filter((e) => e.kind === 'h');
    const vs = this.graph.edges.filter((e) => e.kind === 'v');
    const resample = (e, step) => {
      const pts = e.pts, out = [];
      for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i], b = pts[i + 1];
        const len = Math.hypot(b.x - a.x, b.z - a.z) || 1e-6;
        const m = Math.max(1, Math.ceil(len / step));
        for (let j = 0; j < m; j++) {
          const t = j / m;
          out.push({ x: a.x + (b.x - a.x) * t, z: a.z + (b.z - a.z) * t });
        }
      }
      const last = pts[pts.length - 1];
      out.push({ x: last.x, z: last.z });
      return out;
    };
    const dirAt = (pts, i) => {
      const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
      const dx = b.x - a.x, dz = b.z - a.z, l = Math.hypot(dx, dz) || 1;
      return { x: dx / l, z: dz / l };
    };
    const out = [];
    for (const e of hs) {
      const A = resample(e, 2);
      for (const f of vs) {
        const B = resample(f, 2);
        let bi = 0, bj = 0, bd = Infinity;
        for (let i = 0; i < A.length; i++) {
          const a = A[i];
          for (let j = 0; j < B.length; j++) {
            const b = B[j];
            const d = (a.x - b.x) * (a.x - b.x) + (a.z - b.z) * (a.z - b.z);
            if (d < bd) { bd = d; bi = i; bj = j; }
          }
        }
        if (bd < 20.25) { // closest approach < 4.5m -> streets cross here
          const a = A[bi], b = B[bj];
          out.push({ x: (a.x + b.x) / 2, z: (a.z + b.z) / 2, dH: dirAt(A, bi), dV: dirAt(B, bj) });
        }
      }
    }
    return out;
  }

  _heightAt(x, z) {
    const t = this.world.terrain;
    if (!t) return 0;
    // reuse the shared heightfield via bilinear (mirrors terrain.getHeight)
    const res = t.res, heights = t.heights, n = res + 1;
    const size = t.size, cell = size / res, half = size / 2;
    const gx = clamp((x + half) / cell, 0, res - 1e-4);
    const gz = clamp((z + half) / cell, 0, res - 1e-4);
    const x0 = gx | 0, z0 = gz | 0, fx = gx - x0, fz = gz - z0;
    const i00 = z0 * n + x0, i10 = i00 + 1, i01 = i00 + n, i11 = i01 + 1;
    const a = heights[i00] + (heights[i10] - heights[i00]) * fx;
    const b = heights[i01] + (heights[i11] - heights[i01]) * fx;
    return a + (b - a) * fz;
  }

  // Procedural asphalt + lane markings. v runs along the road (dashes, 12m per
  // repeat), u across. Tire-wear channels down each lane, contact AO at the
  // curbs, fine pores + low-frequency mottling.
  _makeRoadTexture() {
    const T = this.ctx.three;
    const W = 128, H = 256;
    const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
    const g = cv.getContext('2d');
    // asphalt base
    g.fillStyle = '#2e3034'; g.fillRect(0, 0, W, H);
    const img = g.getImageData(0, 0, W, H);
    const d = img.data;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const u = x / W;
        let s = 1;
        // fine pore speckle (grain)
        s *= 0.86 + 0.28 * hash(x * 12.9898 + y * 78.233);
        // low-frequency mottling: aged patches, old repairs
        s *= 0.92 + 0.12 * Math.abs(Math.sin(x * 0.11 + 3.1) * Math.sin(y * 0.07 + 1.7));
        // tire-wear strips: traffic grinds a dark channel down each lane
        const w1 = (u - 0.28) / 0.05, w2 = (u - 0.72) / 0.05;
        const wear = Math.exp(-w1 * w1) + Math.exp(-w2 * w2);
        s *= 1 - 0.22 * wear;
        // contact AO: asphalt darkens against the curb
        const edge = Math.min(u, 1 - u);
        if (edge < 0.07) s *= 1 - 0.15 * (1 - edge / 0.07);
        d[i] *= s; d[i + 1] *= s; d[i + 2] *= s;
      }
    }
    g.putImageData(img, 0, 0);
    // edge lines (solid white)
    g.fillStyle = 'rgba(232,232,235,0.9)';
    g.fillRect(W * 0.065, 0, 4, H); g.fillRect(W * 0.935 - 4, 0, 4, H);
    // centre dashed line (~2.6m dashes on a 6m period)
    g.fillStyle = 'rgba(240,238,220,0.92)';
    for (let y = 8; y < H; y += 132) g.fillRect(W * 0.5 - 2, y, 4, 55);
    const tex = new T.CanvasTexture(cv);
    tex.wrapS = T.RepeatWrapping; tex.wrapT = T.RepeatWrapping;
    tex.colorSpace = T.SRGBColorSpace;
    tex.anisotropy = 1; // >1 bands in SwiftShader
    tex.needsUpdate = true;
    return tex;
  }

  // Procedural sidewalk: light concrete with grain, a dark curb line at the
  // asphalt edge, faint expansion joints every 4m. The asphalt edge sits at
  // u = sw / (2*(hw+sw)) in this ribbon's texture space.
  _makeSidewalkTexture() {
    const T = this.ctx.three;
    const W = 128, H = 256;
    const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
    const g = cv.getContext('2d');
    // light concrete base
    g.fillStyle = '#a9a8a3'; g.fillRect(0, 0, W, H);
    const img = g.getImageData(0, 0, W, H);
    const d = img.data;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const u = x / W;
        let s = 1;
        // concrete speckle
        s *= 0.9 + 0.2 * hash(x * 26.37 + y * 41.13);
        // faint slab mottling
        s *= 0.95 + 0.08 * Math.abs(Math.sin(x * 0.15 + 5.2) * Math.sin(y * 0.09 + 2.3));
        // outer lip shading (tripped-edge darkening at the grass line)
        const edge = Math.min(u, 1 - u);
        if (edge < 0.03) s *= 1 - 0.12 * (1 - edge / 0.03);
        d[i] *= s; d[i + 1] *= s; d[i + 2] *= s;
      }
    }
    g.putImageData(img, 0, 0);
    // dark curb line where the pavement meets the asphalt (both edges)
    const uCurb = this._sw / (2 * (this._hw + this._sw));
    const cw = Math.max(2, Math.round(W * 0.026)); // ~0.34m curb
    g.fillStyle = '#3d4045';
    g.fillRect(W * uCurb - cw / 2, 0, cw, H);
    g.fillRect(W * (1 - uCurb) - cw / 2, 0, cw, H);
    // expansion joints every 4m (85px at 12m per repeat)
    g.fillStyle = 'rgba(30,32,35,0.22)';
    for (let y = 0; y < H; y += 85) g.fillRect(0, y, W, 1);
    const tex = new T.CanvasTexture(cv);
    tex.wrapS = T.RepeatWrapping; tex.wrapT = T.RepeatWrapping;
    tex.colorSpace = T.SRGBColorSpace;
    tex.anisotropy = 1;
    tex.needsUpdate = true;
    return tex;
  }

  // Procedural zebra: white bars thin along the crossing axis (u, the 3m side)
  // and full across the road (v). Gaps are transparent — alphaTest cuts them.
  _makeZebraTexture() {
    const T = this.ctx.three;
    const W = 64, H = 64;
    const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
    const g = cv.getContext('2d');
    g.clearRect(0, 0, W, H);
    g.fillStyle = 'rgba(238,238,230,0.95)';
    for (const x0 of [5, 24, 43]) g.fillRect(x0, 0, 14, H); // three ~0.65m bars
    const tex = new T.CanvasTexture(cv);
    tex.colorSpace = T.SRGBColorSpace;
    tex.anisotropy = 1;
    tex.needsUpdate = true;
    return tex;
  }

  _build() {
    const T = this.ctx.three;
    // 1) asphalt ribbon
    const out = { pos: [], nrm: [], uv: [], idx: [] };
    for (const e of this.graph.edges) ribbon(e.pts, this._hw, out);
    const geo = new T.BufferGeometry();
    geo.setAttribute('position', new T.Float32BufferAttribute(out.pos, 3));
    geo.setAttribute('normal', new T.Float32BufferAttribute(out.nrm, 3));
    geo.setAttribute('uv', new T.Float32BufferAttribute(out.uv, 2));
    geo.setIndex(out.idx);
    geo.computeBoundingSphere();
    this._tex = this._makeRoadTexture();
    const mat = new T.MeshStandardMaterial({ map: this._tex, roughness: 0.92, metalness: 0.0 });
    this._mesh = new T.Mesh(geo, mat);
    this._mesh.receiveShadow = true;

    // 2) sidewalk band: same polylines 0.05m lower, 1.5m wider per side.
    //    The asphalt sits on top, so only the outer band + curb line show.
    const so = { pos: [], nrm: [], uv: [], idx: [] };
    for (const e of this.graph.edges) {
      const pts = e.pts.map((p) => ({ x: p.x, z: p.z, y: p.y - 0.05, _s: p._s }));
      ribbon(pts, this._hw + this._sw, so);
    }
    const sgeo = new T.BufferGeometry();
    sgeo.setAttribute('position', new T.Float32BufferAttribute(so.pos, 3));
    sgeo.setAttribute('normal', new T.Float32BufferAttribute(so.nrm, 3));
    sgeo.setAttribute('uv', new T.Float32BufferAttribute(so.uv, 2));
    sgeo.setIndex(so.idx);
    sgeo.computeBoundingSphere();
    this._sideTex = this._makeSidewalkTexture();
    const smat = new T.MeshStandardMaterial({
      map: this._sideTex, roughness: 0.95, metalness: 0.0,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });
    this._sideMesh = new T.Mesh(sgeo, smat);
    this._sideMesh.receiveShadow = true;

    // 3) zebra crosswalks: one 3m x 10m quad per approach at each grid
    //    intersection, 0.1m above the asphalt (no z-fight), snapped per corner.
    const xo = { pos: [], nrm: [], uv: [], idx: [] };
    for (const ix of this._intersections) {
      for (const dd of [ix.dH, ix.dV]) {
        for (const sgn of [1, -1]) {
          const nx = -dd.z, nz = dd.x;
          const cx = ix.x + sgn * dd.x * 6.5, cz = ix.z + sgn * dd.z * 6.5;
          // a/b are the near cross-section (+n first, matching ribbon winding so
          // the normal is +Y), c/d the far one
          const cs = [
            [cx - 1.5 * dd.x + 5 * nx, cz - 1.5 * dd.z + 5 * nz, 0, 1],
            [cx - 1.5 * dd.x - 5 * nx, cz - 1.5 * dd.z - 5 * nz, 0, 0],
            [cx + 1.5 * dd.x + 5 * nx, cz + 1.5 * dd.z + 5 * nz, 1, 1],
            [cx + 1.5 * dd.x - 5 * nx, cz + 1.5 * dd.z - 5 * nz, 1, 0],
          ];
          const base = xo.pos.length / 3;
          for (const c of cs) {
            xo.pos.push(c[0], this._heightAt(c[0], c[1]) + 0.35, c[1]);
            xo.nrm.push(0, 1, 0);
            xo.uv.push(c[2], c[3]);
          }
          xo.idx.push(base, base + 2, base + 1, base + 1, base + 2, base + 3);
        }
      }
    }
    if (xo.idx.length) {
      const xgeo = new T.BufferGeometry();
      xgeo.setAttribute('position', new T.Float32BufferAttribute(xo.pos, 3));
      xgeo.setAttribute('normal', new T.Float32BufferAttribute(xo.nrm, 3));
      xgeo.setAttribute('uv', new T.Float32BufferAttribute(xo.uv, 2));
      xgeo.setIndex(xo.idx);
      xgeo.computeBoundingSphere();
      this._xwTex = this._makeZebraTexture();
      const xmat = new T.MeshStandardMaterial({ map: this._xwTex, roughness: 0.85, metalness: 0.0, alphaTest: 0.5 });
      this._xwMesh = new T.Mesh(xgeo, xmat);
      this._xwMesh.receiveShadow = true;
    }

    this._root = new T.Group();
    this._root.add(this._mesh, this._sideMesh);
    if (this._xwMesh) this._root.add(this._xwMesh);
  }

  update(dt, world) {}

  // ---- public graph API ------------------------------------------------------
  addEdge(e) {
    if (!e || e.a == null || e.b == null) return;
    const getH = (x, z) => this._heightAt(x, z);
    const pts = [];
    for (let i = 0; i <= 12; i++) {
      const t = i / 12;
      const x = e.a.x + (e.b.x - e.a.x) * t, z = e.a.z + (e.b.z - e.a.z) * t;
      pts.push({ x, z, y: getH(x, z) + 0.25, _s: i });
    }
    this.graph.edges.push({ a: { ...e.a }, b: { ...e.b }, pts, kind: 'a' });
    this.ctx.events.emit('roads:edge', { a: this.graph.edges[this.graph.edges.length - 1].a, b: this.graph.edges[this.graph.edges.length - 1].b });
  }
  samplePoint(edge, t) {
    const pts = edge && edge.pts;
    if (!pts || !pts.length) return { x: 0, y: 0, z: 0 };
    const f = clamp(t, 0, 1) * (pts.length - 1);
    const i = Math.min(pts.length - 2, f | 0), fr = f - i;
    const a = pts[i], b = pts[i + 1];
    return { x: a.x + (b.x - a.x) * fr, y: a.y + (b.y - a.y) * fr, z: a.z + (b.z - a.z) * fr };
  }
  nearestEdge(x, z) {
    let best = null, bd = Infinity;
    for (const e of this.graph.edges) {
      for (let i = 0; i < e.pts.length; i += 2) {
        const p = e.pts[i];
        const d = (p.x - x) ** 2 + (p.z - z) ** 2;
        if (d < bd) { bd = d; best = e; }
      }
    }
    return best;
  }

  showcase(scene, world, ctx) {
    for (const c of [...scene.children]) {
      if (c.isMesh && !this._root.children.includes(c)) scene.remove(c);
    }
    // roads snap to the terrain, so stage the ground underneath them (we own the
    // _root for the duration of the showcase and restore it in dispose).
    const terrain = ctx.registry && ctx.registry.get('terrain');
    if (terrain && terrain._root) {
      terrain._root.removeFromParent();
      scene.add(terrain._root);
      this._terrainRoot = terrain._root;
    }
    scene.add(this._root);
    const env = ctx.registry && ctx.registry.get('environment');
    if (env && env._ibRT) { scene.environment = env._ibRT.texture; scene.environmentIntensity = 0.55; }
    for (const o of this._showExtras) { o.removeFromParent(); if (o.geometry) o.geometry.dispose(); }
    this._showExtras.length = 0;
    const sun = new ctx.three.DirectionalLight(0xfff6ec, 2.2);
    sun.position.set(200, 300, 140); sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera; sc.left = -260; sc.right = 260; sc.top = 260; sc.bottom = -260; sc.near = 20; sc.far = 900; sc.updateProjectionMatrix();
    sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.9;
    scene.add(sun, sun.target);
    this._showExtras.push(sun, sun.target);
  }

  dispose() {
    if (this._root) this._root.removeFromParent();
    for (const o of this._showExtras) { o.removeFromParent(); if (o.geometry) o.geometry.dispose(); }
    this._showExtras.length = 0;
    if (this._terrainRoot) { this.ctx.scene.add(this._terrainRoot); this._terrainRoot = null; } // give terrain back to the live scene
    for (const m of [this._mesh, this._sideMesh, this._xwMesh]) {
      if (m) { m.geometry.dispose(); m.material.dispose(); }
    }
    for (const t of [this._tex, this._sideTex, this._xwTex]) if (t) t.dispose();
    this._mesh = this._tex = this._sideMesh = this._sideTex = this._xwMesh = this._xwTex = this._root = null;
  }

  stats() {
    return {
      drawCalls: 3,
      edges: this.graph.edges.length, nodes: this.graph.nodes.length,
      intersections: this._intersections.length,
      notes: '3 merged ribbons: asphalt+markings (wear/AO), sidewalk+curb, zebra crosswalks',
    };
  }
}
