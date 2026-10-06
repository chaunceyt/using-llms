// tools — the player's interaction layer: pointer picking, ghost previews, and the
// build/zone/road/bulldoze tool state machine (Cities: Skylines style).
//
// Picking: the pointer NDC is unprojected through the camera and the ray is
// MARCHED over the shared heightfield in 2 m steps (up to 600 m). The first sample
// where the ray is at/below the terrain (or within a 1.5 m grazing tolerance) wins;
// a bisection then tightens the crossing to ~2 cm. This avoids mesh-raycast
// internals entirely and is a few dozen bilinear height lookups per frame.
//
// Ghosts: one cheap mesh per tool (zone tile / bulldoze disc / road anchor disc +
// ribbon strip), translucent, depthWrite off, high renderOrder, added to the LIVE
// scene. Only the active tool's ghost is visible. Max 3 draw calls.
//
// Toolbar: a small DOM panel (bottom-left, above the canvas) with the six tools +
// hotkeys. 1/2/3/4 = select/res/com/ind, R = road, B = bulldoze, Esc = select.
//
// Public API: setTool(name), getState(), projectToScreen(x,y,z) (test hook).
// Emits 'tools:activated' {tool} on setTool and 'tools:placed' {tool,x,z,...} per
// apply (each painted tile, each road edge, each bulldoze).
import * as THREE from 'three';

const TOOLS = [
  { id: 'select',   label: 'Select',   kbd: '1', swatch: 'transparent' },
  { id: 'zone-res', label: 'Res',      kbd: '2', swatch: '#56be6e' },
  { id: 'zone-com', label: 'Com',      kbd: '3', swatch: '#4682e6' },
  { id: 'zone-ind', label: 'Ind',      kbd: '4', swatch: '#eb963c' },
  { id: 'road',     label: 'Road',     kbd: 'R', swatch: '#d8d8dc' },
  { id: 'bulldoze', label: 'Bulldoze', kbd: 'B', swatch: '#e0483c' },
];
const ZONE_COLORS = { 'zone-res': 0x56be6e, 'zone-com': 0x4682e6, 'zone-ind': 0xeb963c };

const TSIZE = 8;      // zoning tile size (m) — mirrors the zoning module
const HALF = 140;     // zoning grid span (±140 m)
const N_TILE = 35;    // 35×35 zoning grid

const MARCH_STEP = 2;   // ray-march step (m)
const MARCH_MAX = 600;  // max march distance (m)

const CSS = `
.sk-tools{position:fixed;left:14px;bottom:16px;z-index:30;pointer-events:auto;
  display:flex;flex-direction:column;gap:4px;padding:10px;
  background:rgba(10,16,26,.72);border:1px solid rgba(120,160,220,.22);
  border-radius:10px;backdrop-filter:blur(6px);box-shadow:0 6px 24px rgba(0,0,0,.35);
  font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;color:#eaf2ff;user-select:none;}
.sk-tools-title{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#8fb4e6;margin-bottom:2px;}
.sk-tools-btn{pointer-events:auto;cursor:pointer;display:flex;align-items:center;gap:8px;
  font:inherit;color:#cfdcf0;background:rgba(70,110,170,.18);
  border:1px solid rgba(120,160,220,.18);border-radius:7px;padding:5px 8px;min-width:122px;
  transition:background .12s,border-color .12s;}
.sk-tools-btn:hover{background:rgba(90,140,210,.32);}
.sk-tools-btn.on{background:rgba(90,150,230,.55);border-color:rgba(150,190,255,.6);color:#fff;}
.sk-tools-btn .sw{width:12px;height:12px;border-radius:3px;border:1px solid rgba(255,255,255,.28);flex:none;}
.sk-tools-btn .lbl{flex:1;text-align:left;}
.sk-tools-btn kbd{font:10px/1 inherit;color:#9fb2cc;background:rgba(0,0,0,.3);border-radius:4px;padding:2px 5px;}
`;

export default class Tools {
  name = 'tools';

  constructor() {
    this.ctx = null; this.world = null;
    this.tool = 'select';
    this.anchor = null;      // road start {x,z} (world metres)
    this.lastHit = null;     // last ground hit {x,y,z}
    this._ndc = null;        // pointer NDC (THREE.Vector3) — NEVER mutated by picking
    this._scratch = null;    // ray-march scratch vector
    this._pointerIn = false;
    this._painting = false;
    this._lastPaint = null;  // {tx,tz} last painted tile (drag runs)
    this._ghosts = {};       // id -> mesh
    this._btns = new Map();
    this._root = null; this._style = null;
    this._offs = [];
    // Showcase staging (gauntlet): dedicated ghost/ring/preview meshes in their own
    // group (never the live this._ghosts — update() would hide those), plus the
    // BORROWED terrain/roads _roots that are MOVED into the showcase scene and
    // restored to ctx.scene in dispose().
    this._show = null;           // { outline, volume } — pulse targets
    this._showT = 0; this._showPhase = 0;
    this._showGroup = null; this._showSun = null;
    this._showMeshes = [];       // geometries/materials owned by the staging
    this._showTerrainRoot = null;
    this._showRoadsRoot = null;
  }

  async init(world, ctx) {
    this.ctx = ctx; this.world = world;
    this._ndc = new ctx.three.Vector3(0, 0, 0.5);
    this._scratch = new ctx.three.Vector3();
    this._buildToolbar();
    this._buildGhosts();
    this._bindEvents();
    this.setTool('select');
  }

  _terr() { return this.ctx.registry && this.ctx.registry.get('terrain'); }

  // ---- toolbar (DOM) -----------------------------------------------------------
  _buildToolbar() {
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);
    this._style = style;

    const root = document.createElement('div');
    root.className = 'sk-tools';
    const title = document.createElement('div');
    title.className = 'sk-tools-title';
    title.textContent = 'Tools';
    root.appendChild(title);

    for (const t of TOOLS) {
      const b = document.createElement('button');
      b.className = 'sk-tools-btn';
      b.dataset.tool = t.id;
      b.innerHTML =
        `<span class="sw" style="background:${t.swatch}"></span>` +
        `<span class="lbl">${t.label}</span><kbd>${t.kbd}</kbd>`;
      b.addEventListener('click', () => this.setTool(t.id));
      root.appendChild(b);
      this._btns.set(t.id, b);
    }
    document.body.appendChild(root);
    this._root = root;
  }

  // ---- ghost meshes -------------------------------------------------------------
  _buildGhosts() {
    const T = this.ctx.three;
    const scene = this.ctx.scene;
    const ghost = (geo, color, opacity) => {
      const m = new T.Mesh(geo, new T.MeshBasicMaterial({
        color, transparent: true, opacity, depthWrite: false, side: T.DoubleSide,
      }));
      m.renderOrder = 10;
      m.visible = false;
      scene.add(m);
      return m;
    };

    const zoneGeo = new T.PlaneGeometry(TSIZE, TSIZE);
    zoneGeo.rotateX(-Math.PI / 2);
    this._ghosts.zone = ghost(zoneGeo, ZONE_COLORS['zone-res'], 0.55);

    const bdGeo = new T.CircleGeometry(10, 40);
    bdGeo.rotateX(-Math.PI / 2);
    this._ghosts.bulldoze = ghost(bdGeo, 0xe0483c, 0.32);

    const anchorGeo = new T.CircleGeometry(2.5, 24);
    anchorGeo.rotateX(-Math.PI / 2);
    this._ghosts.roadAnchor = ghost(anchorGeo, 0xf2e9c8, 0.85);

    // road ribbon: 4-vertex strip, vertices rewritten as the cursor moves
    const ribbonGeo = new T.BufferGeometry();
    ribbonGeo.setAttribute('position', new T.BufferAttribute(new Float32Array(12), 3));
    ribbonGeo.setIndex([0, 1, 2, 0, 2, 3]);
    const ribbon = ghost(ribbonGeo, 0xf2e9c8, 0.75);
    ribbon.frustumCulled = false; // vertices rewritten per frame
    this._ghosts.roadRibbon = ribbon;
  }

  _setRibbon(ax, az, bx, bz) {
    const pos = this._ghosts.roadRibbon.geometry.getAttribute('position');
    const g = 1.5;
    let dx = bx - ax, dz = bz - az;
    const l = Math.hypot(dx, dz) || 1;
    dx /= l; dz /= l;
    const nx = -dz * g, nz = dx * g;
    const t = this._terr();
    const ay = t ? t.getHeight(ax, az) + 0.5 : 0.5;
    const by = t ? t.getHeight(bx, bz) + 0.5 : 0.5;
    pos.setXYZ(0, ax + nx, ay, az + nz);
    pos.setXYZ(1, ax - nx, ay, az - nz);
    pos.setXYZ(2, bx + nx, by, bz + nz);
    pos.setXYZ(3, bx - nx, by, bz - nz);
    pos.needsUpdate = true;
  }

  // ---- input --------------------------------------------------------------------
  _bindEvents() {
    const cv = this.ctx.canvas;
    const on = (obj, type, fn) => {
      obj.addEventListener(type, fn);
      this._offs.push(() => obj.removeEventListener(type, fn));
    };
    on(cv, 'pointermove', (e) => this._onPointer(e, false));
    on(cv, 'pointerdown', (e) => this._onPointer(e, true));
    on(window, 'pointerup', () => { this._painting = false; this._lastPaint = null; });
    on(cv, 'pointerleave', () => { this._pointerIn = false; this._hideGhosts(); });
    on(window, 'keydown', (e) => this._onKey(e));
  }

  _onPointer(e, isDown) {
    const r = this.ctx.canvas.getBoundingClientRect();
    if (!r.width || !r.height) return;
    this._ndc.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    this._ndc.y = -((e.clientY - r.top) / r.height) * 2 + 1;
    this._pointerIn = true;
    if (!isDown || e.button !== 0) return;
    this._apply();
  }

  _onKey(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return; // never swallow browser shortcuts
    let tool = null;
    switch (e.key) {
      case '1': tool = 'select'; break;
      case '2': tool = 'zone-res'; break;
      case '3': tool = 'zone-com'; break;
      case '4': tool = 'zone-ind'; break;
      case 'r': case 'R': tool = 'road'; break;
      case 'b': case 'B': tool = 'bulldoze'; break;
      case 'Escape': tool = 'select'; break;
      default: return;
    }
    e.preventDefault();
    this.setTool(tool);
  }

  // ---- applying an action at the current ground hit ------------------------------
  _apply() {
    const hit = this._groundHit();
    if (!hit) return;
    this.lastHit = hit;
    const tool = this.tool;
    if (tool === 'select') return;

    if (tool.startsWith('zone-')) {
      this._paintAt(hit);
      this._painting = true;
      return;
    }
    if (tool === 'road') {
      if (!this.anchor) {
        this.anchor = { x: hit.x, z: hit.z };
        this.ctx.events.emit('tools:placed', { tool: 'road', phase: 'anchor', x: hit.x, z: hit.z });
      } else {
        const a = this.anchor;
        this.anchor = null;
        const roads = this.ctx.registry.get('roads');
        if (roads && typeof roads.addEdge === 'function') {
          roads.addEdge({ a: { x: a.x, z: a.z }, b: { x: hit.x, z: hit.z } });
        }
        this.ctx.events.emit('tools:placed', { tool: 'road', a: { x: a.x, z: a.z }, b: { x: hit.x, z: hit.z } });
      }
      return;
    }
    if (tool === 'bulldoze') {
      const bld = this.ctx.registry.get('buildings');
      if (bld && typeof bld.removeAt === 'function') {
        try { bld.removeAt(hit.x, hit.z); } catch { /* best effort */ }
      }
      this.ctx.events.emit('tools:placed', { tool: 'bulldoze', x: hit.x, z: hit.z });
    }
  }

  // paint the tile under the hit; if a drag is running, walk the (staircase) run
  // from the last painted tile so fast drags leave no gaps
  _paintAt(hit) {
    const tx = Math.floor((hit.x + HALF) / TSIZE);
    const tz = Math.floor((hit.z + HALF) / TSIZE);
    if (tx < 0 || tx >= N_TILE || tz < 0 || tz >= N_TILE) { this._lastPaint = null; return; }
    const use = toolUse(this.tool);
    const zoning = this.ctx.registry.get('zoning');
    if (!zoning || typeof zoning.setZone !== 'function') return;
    if (this._lastPaint && (this._lastPaint.tx !== tx || this._lastPaint.tz !== tz)) {
      let cx = this._lastPaint.tx, cz = this._lastPaint.tz;
      const dx = Math.sign(tx - cx), dz = Math.sign(tz - cz);
      while (cx !== tx || cz !== tz) {
        if (cx !== tx) cx += dx; else cz += dz;
        zoning.setZone({ tx: cx, tz: cz }, use, 3);
      }
    }
    zoning.setZone({ tx, tz }, use, 3);
    this._lastPaint = { tx, tz };
  }

  // ---- picking: ray-march the pointer ray against the heightfield -----------------
  _groundHit() {
    const cam = this.ctx.camera;
    const terr = this._terr();
    if (!terr) return null;
    // copy into the scratch vector: unproject/sub mutate in place, and the
    // stored NDC must survive every call (it is re-read each frame in update)
    const v = this._scratch.set(this._ndc.x, this._ndc.y, 0.5).unproject(cam);
    const dir = v.sub(cam.position);
    const dl = dir.length();
    if (dl < 1e-6) return null;
    dir.multiplyScalar(1 / dl);
    const o = cam.position;
    const wn = this.world.terrain.waterLevel;
    const getH = (t) => terr.getHeight(o.x + dir.x * t, o.z + dir.z * t);

    let t0 = 0;
    let f0 = o.y - getH(0); // ray height minus terrain height (always > 0 when skipped)
    for (let t = MARCH_STEP; t <= MARCH_MAX; t += MARCH_STEP) {
      const h = getH(t);
      if (h <= wn - 0.5) { t0 = t; f0 = o.y + dir.y * t - h; continue; } // open sea: no hit
      const f = o.y + dir.y * t - h;
      if (f > 1.5) { t0 = t; f0 = f; continue; }                          // clearly above ground
      let tt = t;
      if (f <= 0 && f0 > 0) {
        // sign change on [t0, t]: bisect to tighten the crossing (~2 cm after 14 iters)
        let lo = t0, hi = t;
        for (let i = 0; i < 14 && hi - lo > 0.02; i++) {
          const tm = (lo + hi) * 0.5;
          if (o.y + dir.y * tm - getH(tm) > 0) lo = tm; else hi = tm;
        }
        tt = hi;
      }
      const hx = o.x + dir.x * tt, hz = o.z + dir.z * tt;
      return { x: hx, y: terr.getHeight(hx, hz), z: hz };
    }
    return null;
  }

  // ---- public API ------------------------------------------------------------------
  setTool(name) {
    if (!TOOLS.some((t) => t.id === name)) return;
    this.tool = name;
    if (name !== 'road') this.anchor = null;
    this._painting = false;
    this._lastPaint = null;
    for (const t of TOOLS) this._btns.get(t.id)?.classList.toggle('on', t.id === name);
    if (this.ctx && this.ctx.canvas) this.ctx.canvas.style.cursor = name === 'select' ? '' : 'crosshair';
    this._hideGhosts();
    this.ctx.events.emit('tools:activated', { tool: name });
  }

  getState() {
    return {
      tool: this.tool,
      anchor: this.anchor ? { ...this.anchor } : null,
      lastHit: this.lastHit ? { ...this.lastHit } : null,
    };
  }

  // project a world point to canvas pixel coords (verification hook for probe.mjs)
  projectToScreen(x, y, z) {
    const T = this.ctx.three;
    const v = new T.Vector3(x, y, z).project(this.ctx.camera);
    const w = this.ctx.canvas.clientWidth, h = this.ctx.canvas.clientHeight;
    return { sx: Math.round((v.x + 1) * 0.5 * w), sy: Math.round((1 - v.y) * 0.5 * h) };
  }

  // ---- per-frame --------------------------------------------------------------------
  update(dt, world) {
    // Showcase ghost pulse: opacity/scale oscillate but stay fully legible at
    // every phase (the gauntlet frame is static, so nothing may fade out).
    if (this._show) {
      this._showT += dt;
      const p = 0.5 + 0.5 * Math.sin(this._showT * 2.2 + this._showPhase);
      this._show.outline.material.opacity = 0.6 + 0.35 * p;
      const s = 1 + 0.045 * p;
      this._show.outline.scale.set(s, 1, s); // local-vertex loop, scales about its centre
      this._show.volume.material.opacity = 0.24 + 0.14 * p;
    }
    const g = this._ghosts;
    if (!g.zone) return;
    if (this.tool === 'select' || !this._pointerIn) { this._hideGhosts(); return; }
    const hit = this._groundHit();
    this.lastHit = hit;
    if (!hit) { this._hideGhosts(); return; }
    const t = this._terr();

    if (this.tool.startsWith('zone-')) {
      if (this._painting) this._paintAt(hit); // drag paint: fill newly entered tiles
      const tx = Math.floor((hit.x + HALF) / TSIZE);
      const tz = Math.floor((hit.z + HALF) / TSIZE);
      if (tx < 0 || tx >= N_TILE || tz < 0 || tz >= N_TILE) { g.zone.visible = false; return; }
      const cx = -HALF + tx * TSIZE + TSIZE / 2;
      const cz = -HALF + tz * TSIZE + TSIZE / 2;
      g.zone.material.color.setHex(ZONE_COLORS[this.tool]);
      g.zone.position.set(cx, t.getHeight(cx, cz) + 0.5, cz);
      g.zone.visible = true;
    } else if (this.tool === 'road') {
      const px = this.anchor ? this.anchor.x : hit.x;
      const pz = this.anchor ? this.anchor.z : hit.z;
      g.roadAnchor.position.set(px, t.getHeight(px, pz) + 0.45, pz);
      g.roadAnchor.visible = true;
      if (this.anchor) {
        this._setRibbon(this.anchor.x, this.anchor.z, hit.x, hit.z);
        g.roadRibbon.visible = true;
      } else {
        g.roadRibbon.visible = false;
      }
    } else if (this.tool === 'bulldoze') {
      g.bulldoze.position.set(hit.x, t.getHeight(hit.x, hit.z) + 0.4, hit.z);
      g.bulldoze.visible = true;
    }
  }

  _hideGhosts() { for (const k in this._ghosts) this._ghosts[k].visible = false; }

  // ---- showcase -----------------------------------------------------------------------
  // Tools act on the LIVE scene; the gauntlet frame stages a full placement flow in a
  // clean scene: the REAL terrain + road grid (borrowed _roots, MOVED not cloned,
  // restored in dispose) and a residential lot being placed on it — translucent
  // footprint + building volume with a pulsing outline, a snap range ring, a
  // dashed road-drag preview to the nearest street, and snap marks at the lot
  // corners and the road junction. Dedicated meshes (not this._ghosts) so the
  // live per-frame ghost logic can never hide the staged pose.
  showcase(scene, world, ctx) {
    const T = ctx.three;
    this._clearShowcase();

    // 1) CITY CONTEXT — move the terrain + roads roots into the showcase scene.
    const terrain = ctx.registry && ctx.registry.get('terrain');
    if (terrain && terrain._root) {
      terrain._root.removeFromParent();
      scene.add(terrain._root);
      this._showTerrainRoot = terrain._root;
    }
    const roads = ctx.registry && ctx.registry.get('roads');
    if (roads && roads._root) {
      roads._root.removeFromParent();
      scene.add(roads._root);
      this._showRoadsRoot = roads._root;
    }
    // The harness's flat staging ground would sit on top of the terrain — the
    // borrowed roots are Groups, so sweeping isMesh children only hits it.
    for (const c of [...scene.children]) if (c.isMesh) scene.remove(c);

    // Day IBL + warm sun (same pattern the terrain/roads showcases use) so the
    // asphalt and grass read with real shading instead of a flat hemisphere fill.
    const env = ctx.registry && ctx.registry.get('environment');
    if (env && env._ibRT) { scene.environment = env._ibRT.texture; scene.environmentIntensity = 0.55; }
    const sun = new T.DirectionalLight(0xfff6ec, 2.2);
    sun.position.set(200, 300, 140); sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = -260; sc.right = 260; sc.top = 260; sc.bottom = -260; sc.near = 20; sc.far = 900;
    sc.updateProjectionMatrix();
    sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.9;
    scene.add(sun, sun.target);
    this._showSun = sun;

    const rng = ctx.rng.fork('tools-showcase');
    this._showPhase = rng.next() * Math.PI * 2;
    this._showT = 0;

    const H = (x, z) => (terrain ? terrain.getHeight(x, z) : 0);
    const grp = new T.Group();
    this._showGroup = grp;
    const mat = (color, opacity) => new T.MeshBasicMaterial({
      color, transparent: true, opacity, depthWrite: false, side: T.DoubleSide,
    });
    const add = (obj, order) => { obj.renderOrder = order; grp.add(obj); this._showMeshes.push(obj); return obj; };

    // 2) PLACEMENT GHOST — a 16×16 m residential lot (2×2 zoning tiles) snapped to
    // the 8 m grid, centred where the street camera reads it clearly.
    const GX = 16, GZ = -16, FS = 16, half = FS / 2;
    const gy = H(GX, GZ);
    const fill = add(new T.Mesh(new T.PlaneGeometry(FS, FS).rotateX(-Math.PI / 2), mat(0x2f9e54, 0.55)), 10);
    fill.position.set(GX, gy + 0.55, GZ);
    const volume = add(new T.Mesh(new T.BoxGeometry(FS - 4, 11, FS - 4), mat(0x46b873, 0.4)), 10);
    volume.position.set(GX, gy + 5.55, GZ);
    // flat border frame (4 quads) — a 1px line is invisible at street distance.
    // LOCAL vertices, centred on the origin, so the pulse scales about the centre.
    const frameGeo = (S, w) => {
      const o = S / 2, i = o - w;
      const p = [], ix = [];
      const quad = (a, b, c, d) => {
        const base = p.length / 3;
        p.push(a[0], 0, a[1], b[0], 0, b[1], c[0], 0, c[1], d[0], 0, d[1]);
        ix.push(base, base + 1, base + 2, base, base + 2, base + 3);
      };
      quad([-o, -o], [o, -o], [o, -i], [-o, -i]);   // south
      quad([-o, i], [o, i], [o, o], [-o, o]);       // north
      quad([-o, -i], [-o, i], [-i, i], [-i, -i]);   // west
      quad([i, -i], [i, i], [o, i], [o, -i]);       // east
      const g = new T.BufferGeometry();
      g.setAttribute('position', new T.Float32BufferAttribute(p, 3));
      g.setIndex(ix);
      return g;
    };
    const outline = add(new T.Mesh(frameGeo(FS, 0.7), new T.MeshBasicMaterial({
      color: 0xeafff0, transparent: true, opacity: 0.95, depthWrite: false, side: T.DoubleSide,
    })), 11);
    outline.position.set(GX, gy + 0.78, GZ);
    const topFrame = add(new T.Mesh(frameGeo(FS - 4, 0.5), new T.MeshBasicMaterial({
      color: 0xbaffc8, transparent: true, opacity: 0.75, depthWrite: false, side: T.DoubleSide,
    })), 11);
    topFrame.position.set(GX, gy + 11.1, GZ);
    this._show = { outline, volume };

    // 3) RANGE RING — placement/snap range around the lot (bright rim + faint wash).
    const ring = add(new T.Mesh(new T.RingGeometry(17.2, 19.2, 56).rotateX(-Math.PI / 2), mat(0xcfeaff, 0.9)), 9);
    ring.position.set(GX, gy + 0.42, GZ);
    const wash = add(new T.Mesh(new T.CircleGeometry(19, 56).rotateX(-Math.PI / 2), mat(0x9fd8ff, 0.06)), 8);
    wash.position.set(GX, gy + 0.3, GZ);

    // 4) ROAD-DRAG PREVIEW — translucent ribbon + dashed centreline from the lot's
    // near corner to the nearest point on the REAL road network (the snap target).
    const p0 = { x: GX + half, z: GZ - 6 }; // lot south-east corner area
    const want = { x: 48, z: -34 };
    let snap = null, bd = Infinity;
    if (roads) for (const e of roads.graph.edges) for (const p of e.pts) {
      const d = (p.x - want.x) ** 2 + (p.z - want.z) ** 2;
      if (d < bd) { bd = d; snap = p; }
    }
    if (!snap) snap = { x: want.x, z: want.z };
    const p0y = H(p0.x, p0.z) + 0.5, s1y = H(snap.x, snap.z) + 0.5;
    {
      let dx = snap.x - p0.x, dz = snap.z - p0.z;
      const l = Math.hypot(dx, dz) || 1; dx /= l; dz /= l;
      const nx = -dz * 1.6, nz = dx * 1.6;
      const rGeo = new T.BufferGeometry();
      rGeo.setAttribute('position', new T.Float32BufferAttribute([
        p0.x + nx, p0y, p0.z + nz,
        p0.x - nx, p0y, p0.z - nz,
        snap.x + nx, s1y, snap.z + nz,
        snap.x - nx, s1y, snap.z - nz,
      ], 3));
      rGeo.setIndex([0, 1, 2, 0, 2, 3]);
      add(new T.Mesh(rGeo, mat(0xf2e9c8, 0.85)), 10);
    }
    const lGeo = new T.BufferGeometry().setFromPoints([
      new T.Vector3(p0.x, p0y + 0.25, p0.z),
      new T.Vector3(snap.x, s1y + 0.25, snap.z),
    ]);
    const lMat = new T.LineDashedMaterial({
      color: 0xffffff, dashSize: 2.2, gapSize: 1.4, transparent: true, opacity: 0.95,
    });
    if (typeof lMat.dashOffset === 'number') lMat.dashOffset = rng.next() * 3.6;
    const dash = add(new T.Line(lGeo, lMat), 11);
    dash.computeLineDistances();

    // 5) SNAP INDICATORS — corner ticks where the footprint meets the 8 m grid,
    // a start dot at the drag anchor, and a junction marker on the road.
    {
      const tp = []; const tIdx = [];
      for (const [cx, cz] of [[GX - half, GZ - half], [GX + half, GZ - half], [GX - half, GZ + half], [GX + half, GZ + half]]) {
        const k = 1.6, y = gy + 0.66;
        const b = tp.length / 3;
        tp.push(cx - k, y, cz - k,  cx + k, y, cz - k,  cx + k, y, cz + k,  cx - k, y, cz + k);
        tIdx.push(b, b + 1, b + 2,  b, b + 2, b + 3);
      }
      const tGeo = new T.BufferGeometry();
      tGeo.setAttribute('position', new T.Float32BufferAttribute(tp, 3));
      tGeo.setIndex(tIdx);
      add(new T.Mesh(tGeo, mat(0xffffff, 1)), 11);
    }
    const start = add(new T.Mesh(new T.CircleGeometry(1.1, 20).rotateX(-Math.PI / 2), mat(0xffffff, 1)), 11);
    start.position.set(p0.x, p0y + 0.3, p0.z);
    const jRing = add(new T.Mesh(new T.RingGeometry(2.0, 3.0, 28).rotateX(-Math.PI / 2), mat(0xffc23d, 1)), 12);
    jRing.position.set(snap.x, s1y + 0.3, snap.z);
    const jDot = add(new T.Mesh(new T.CircleGeometry(1.0, 20).rotateX(-Math.PI / 2), mat(0xffffff, 1)), 12);
    jDot.position.set(snap.x, s1y + 0.32, snap.z);

    scene.add(grp);
  }

  // Tear down staged showcase objects (group + owned geometries/materials + sun),
  // and give the borrowed terrain/roads roots back to the live scene.
  _clearShowcase() {
    if (this._showGroup) {
      this._showGroup.removeFromParent();
      for (const m of this._showMeshes) {
        if (m.geometry) m.geometry.dispose();
        if (m.material) m.material.dispose();
      }
      this._showGroup = null;
      this._showMeshes = [];
    }
    if (this._showSun) {
      this._showSun.removeFromParent();
      if (this._showSun.target) this._showSun.target.removeFromParent();
      this._showSun = null;
    }
    this._show = null; this._showT = 0;
  }

  dispose() {
    for (const off of this._offs) off();
    this._offs = [];
    if (this._style) { this._style.remove(); this._style = null; }
    if (this._root) { this._root.remove(); this._root = null; }
    this._btns.clear();
    for (const m of Object.values(this._ghosts)) {
      if (!m) continue;
      m.removeFromParent();
      m.geometry.dispose();
      m.material.dispose();
    }
    this._ghosts = {};
    this._clearShowcase();
    // restore the borrowed _roots to the live scene (scene-ownership rule)
    if (this._showTerrainRoot && this.ctx) { this.ctx.scene.add(this._showTerrainRoot); this._showTerrainRoot = null; }
    if (this._showRoadsRoot && this.ctx) { this.ctx.scene.add(this._showRoadsRoot); this._showRoadsRoot = null; }
    this.tool = 'select'; this.anchor = null; this.lastHit = null;
    if (this.ctx && this.ctx.canvas) this.ctx.canvas.style.cursor = '';
  }

  stats() {
    let n = 0;
    for (const k in this._ghosts) if (this._ghosts[k] && this._ghosts[k].visible) n++;
    return {
      drawCalls: n,
      notes: `active: ${this.tool} · ${n} ghost mesh(es) · ray-march picking (2 m steps, bisection refine) · DOM toolbar · showcase: real terrain+roads, lot ghost, range ring, road-drag, snap marks`,
    };
  }
}

function toolUse(tool) {
  return tool === 'zone-res' ? 'residential' : tool === 'zone-com' ? 'commercial' : 'industrial';
}
