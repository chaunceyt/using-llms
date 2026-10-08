// =============================================================================
// buildings — procedural + merged city blocks (Skylines)
//
// Builds a dense, AAA-photographic downtown from zoned cells (`world.zones`) or,
// when zoning has not landed yet, from a deterministic default master plan so the
// module showcases on its own. Buildings sit on terrain via world.heightAt(x,z).
//
// Visual approach:
//   * Procedural PBR facades (albedo / normal / roughness / AO canvas tiles) per
//     architectural style — commercial glass towers, residential masonry slabs,
//     industrial corrugated warehouses.
//   * Every building is real geometry with its own baked window-grid UVs, roof
//     parapet + rooftop AC units, subtle vertex tint so blocks are never cloned.
//   * All wall geometry is MERGED per (style × lit-window-variant) into a handful
//     of BufferGeometries; roofs merge into one. => ~10-12 draw calls total.
//   * Day/night: emissive "lit window" maps switch on after dusk / before dawn so
//     the city glows warm at night (the living-city beat).
//
// Determinism: every layout/height/tint/lit decision draws from world.rng (seeded),
// never Math.random. Textures use an independent integer-hash seeded by style so
// they don't depend on rng consumption order.
//
// Performance: ~500 draw-call budget for the module; this stays around a dozen.
// =============================================================================
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { bus, mulberry32 } from '../core/index.js';

export const id = 'buildings';

// ---------------------------------------------------------------------------
// Deterministic integer hash -> [0,1). Independent of world.rng draw order.
// ---------------------------------------------------------------------------
function hash2(seed, x, y) {
  let n = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263) + Math.imul(seed | 0, 1013904223)) | 0;
  n = Math.imul(n ^ (n >>> 13), 1274126177);
  return ((((n ^ (n >>> 16)) >>> 0) % 100000) / 100000);
}
function hash3(seed, x, y, z) {
  let n = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263) + Math.imul(z | 0, 1013904223) + Math.imul(seed | 0, 1442695041)) | 0;
  n = Math.imul(n ^ (n >>> 15), 2246822519);
  return ((((n ^ (n >>> 13)) >>> 0) % 100000) / 100000);
}

const clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerp = (a, b, t) => a + (b - a) * t;

// world metres spanned by one facade tile horizontally; each tile is one floor.
const FACADE_TILE_W = 5.0;

const FLOOR_H = {
  commercial: 4.0,
  residential: 3.1,
  industrial: 5.0,
};

const NIGHT_LIT = 3.0;         // emissive intensity at full night
const LIT_VARIANTS = 3;        // distinct warm-window patterns per style

const state = {
  materials: [],     // all wall materials (toggle emissive by time of day)
  nightFactor: -1,
  built: false,
};

// ---------------------------------------------------------------------------
// Canvas helpers
// ---------------------------------------------------------------------------
function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return { c, ctx: c.getContext('2d') };
}
function toTex(canvas, sRGB) {
  const t = new THREE.CanvasTexture(canvas);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 4;
  t.colorSpace = sRGB ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  return t;
}

// Sobel-derive a tangent-space normal canvas from a greyscale height canvas.
function sobelNormal(heightCanvas) {
  const W = heightCanvas.width, H = heightCanvas.height;
  const nc = makeCanvas(W, H);
  const ctx = nc.ctx;
  const src = heightCanvas.getContext('2d').getImageData(0, 0, W, H).data;
  const out = ctx.createImageData(W, H);
  const strength = 2.4;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = (y * W + x) * 4;
      const Xm = src[Math.max(x - 1, 0) * 4 + y * W * 4];
      const Xp = src[Math.min(x + 1, W - 1) * 4 + y * W * 4];
      const Ym = src[x * 4 + Math.max(y - 1, 0) * W * 4];
      const Yp = src[x * 4 + Math.min(y + 1, H - 1) * W * 4];
      const gx = (Xp - Xm) / 255;
      const gy = (Yp - Ym) / 255;
      let nx = -gx * strength;
      let ny = -gy * strength;
      let nz = 1;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      nx /= len; ny /= len; nz /= len;
      out.data[idx] = Math.round((nx * 0.5 + 0.5) * 255);
      out.data[idx + 1] = Math.round((ny * 0.5 + 0.5) * 255);
      out.data[idx + 2] = Math.round(nz * 255);
      out.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(out, 0, 0);
  return nc.c;
}

// ---------------------------------------------------------------------------
// Facade tile generators. Each style produces albedo/normal/roughness/AO plus a
// few emissive "lit window" variants. Tiles are ONE floor tall (full V span) and
// FACADE_TILE_W wide; walls bake UV so the grid tiles per floor & column.
// ---------------------------------------------------------------------------
function buildStyleTiles(seed, style) {
  const W = 256, H = 512;
  const albedo = makeCanvas(W, H);
  const heightC = makeCanvas(W, H);   // used for normal
  const roughC = makeCanvas(W, H);    // green channel carries roughness
  const aoC = makeCanvas(W, H);       // red channel carries AO

  const cfg = {
    commercial: {
      wall: [0x8fb4c9, 0xa7c6d8],        // cool glass tint range
      glass: [0xa7ccdc, 0xc3e0ec],
      frame: [0xb8ccd8, 0xcbdbe4],
      spandrel: [0x6f8fa3, 0x7fa3b6],
      windowCols: 8,
      glassRough: 0.12, wallRough: 0.55,
      metalness: 0.25,
      litWarm: true,
    },
    residential: {
      wall: [0xc9b093, 0xb98a6f],        // beige / brick range
      frame: [0x8c7458, 0x7d5a42],
      glass: [0x33424f, 0x42505c],
      windowCols: 4,
      glassRough: 0.28, wallRough: 0.86,
      metalness: 0.04,
      litWarm: true,
    },
    industrial: {
      wall: [0x8f9397, 0xa6a9ac],        // corrugated steel range
      frame: [0x6c7074, 0x54575b],
      glass: [0x2c343b, 0x3a424a],
      windowCols: 6,
      glassRough: 0.4, wallRough: 0.7,
      metalness: 0.78,
      litWarm: false,
    },
  }[style];

  const alb = albedo.ctx, hc = heightC.ctx, rc = roughC.ctx, ac = aoC.ctx;
  const wallCol = new THREE.Color().setHex(cfg.wall[0]).lerp(new THREE.Color().setHex(cfg.wall[1]), 0.5);
  const glassCol = cfg.glass ? new THREE.Color().setHex(cfg.glass[0]) : null;
  const frameCol = new THREE.Color().setHex(cfg.frame[0]);
  const spandrelCol = cfg.spandrel ? new THREE.Color().setHex(cfg.spandrel[0]) : null;

  // ---- base wall + noise --------------------------------------------------
  // Write straight into ImageData buffers (single putImageData each) instead of
  // per-pixel fillRect — ~10x faster at init.
  const ia = albedo.ctx.getImageData(0, 0, W, H);
  const ih = heightC.ctx.getImageData(0, 0, W, H);
  const ir = roughC.ctx.getImageData(0, 0, W, H);
  const iA = aoC.ctx.getImageData(0, 0, W, H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = (y * W + x) * 4;
      const n = hash2(seed, x, y);
      let r = wallCol.r + (n - 0.5) * 0.10;
      // brick courses for masonry
      if (style === 'residential' && ((y >> 4) % 2) === 1) r *= 0.96;
      let g = wallCol.g, b = wallCol.b;
      if (style === 'commercial') {
        const vf = 1 - (y / H);
        g += vf * 0.06; b += vf * 0.09;
      }
      ia.data[idx] = Math.round(r * 255);
      ia.data[idx + 1] = Math.round(g * 255);
      ia.data[idx + 2] = Math.round(b * 255);
      ia.data[idx + 3] = 255;

      // height (relief) base
      let hv = style === 'industrial'
        ? 118 + Math.sin(((x % 16) / 16) * Math.PI) * 34 + (n - 0.5) * 8
        : 126 + (n - 0.5) * 12;
      ih.data[idx] = ih.data[idx + 1] = ih.data[idx + 2] = hv | 0;
      ih.data[idx + 3] = 255;

      // roughness (green channel)
      let roughV = cfg.wallRough * 255;
      if (style === 'residential') roughV += Math.sin(((x + y) % 7) / 7 * Math.PI * 2) * 14;
      ir.data[idx + 1] = Math.round(roughV);
      ir.data[idx + 3] = 255;

      // AO (red channel): darken toward the top spandrel area
      let ao = 200 - Math.max(0, ((y / H) - 0.88)) * 120;
      iA.data[idx] = Math.round(ao);
      iA.data[idx + 3] = 255;
    }
  }
  albedo.ctx.putImageData(ia, 0, 0);
  heightC.ctx.putImageData(ih, 0, 0);
  roughC.ctx.putImageData(ir, 0, 0);
  aoC.ctx.putImageData(iA, 0, 0);

  // ---- windows --------------------------------------------------------------
  // Compute window column rects (normalized), shared by albedo + height + lit.
  const cols = cfg.windowCols;
  const winRects = [];
  for (let c = 0; c < cols; c++) {
    let u0, u1, v0, v1;
    if (style === 'commercial') {
      // full-height curtain wall columns separated by mullions
      const colW = 0.11;
      u0 = c / cols + 0.008; u1 = c / cols + colW - 0.006;
      v0 = 0.06; v1 = 0.92;
    } else if (style === 'residential') {
      const span = 0.15;
      u0 = 0.10 + c * 0.20; u1 = u0 + span;
      v0 = 0.16; v1 = 0.60;
    } else { // industrial: high clerestory windows
      const span = 0.11;
      u0 = 0.08 + c * 0.15; u1 = u0 + span;
      v0 = 0.62; v1 = 0.84;
    }
    winRects.push({ u0, u1, v0, v1 });
  }

  for (const wr of winRects) {
    const px0 = Math.round(wr.u0 * W), px1 = Math.round(wr.u1 * W);
    const py0 = Math.round((1 - wr.v1) * H), py1 = Math.round((1 - wr.v0) * H); // canvas y down
    const wpx = Math.max(2, px1 - px0), hpx = Math.max(2, py1 - py0);

    if (style === 'commercial') {
      // glass pane with light spandrel band at base of the column
      alb.fillStyle = `rgb(${(glassCol.r * 255) | 0},${(glassCol.g * 255) | 0},${(glassCol.b * 255) | 0})`;
      alb.fillRect(px0, py0, wpx, hpx);
      // mullion frame
      alb.strokeStyle = `rgb(${(frameCol.r * 255) | 0},${(frameCol.g * 255) | 0},${(frameCol.b * 255) | 0})`;
      alb.lineWidth = 2;
      alb.strokeRect(px0, py0, wpx, hpx);
      if (spandrelCol && wr.v1 > 0.5) {
        alb.fillStyle = `rgb(${(spandrelCol.r * 255) | 0},${(spandrelCol.g * 255) | 0},${(spandrelCol.b * 255) | 0})`;
        alb.fillRect(px0, py1 - Math.round(hpx * 0.28), wpx, Math.round(hpx * 0.28));
      }
    } else {
      // punched window with sill
      alb.fillStyle = `rgb(${(glassCol.r * 255) | 0},${(glassCol.g * 255) | 0},${(glassCol.b * 255) | 0})`;
      alb.fillRect(px0, py0, wpx, hpx);
      alb.strokeStyle = `rgb(${(frameCol.r * 255) | 0},${(frameCol.g * 255) | 0},${(frameCol.b * 255) | 0})`;
      alb.lineWidth = 3;
      alb.strokeRect(px0 - 2, py0 - 2, wpx + 4, hpx + 4);
      // sill highlight
      alb.fillStyle = 'rgba(255,255,255,0.35)';
      alb.fillRect(px0 - 1, py1 - 1, wpx + 2, 2);
    }

    // height relief: recessed glass, raised frame ring
    hc.fillStyle = 'rgb(110,110,110)';
    hc.fillRect(px0 + 3, py0 + 3, Math.max(1, wpx - 6), Math.max(1, hpx - 6));
    hc.strokeStyle = 'rgb(210,210,210)';
    hc.lineWidth = 3;
    hc.strokeRect(px0, py0, wpx, hpx);

    // roughness: glass low (green)
    rc.fillStyle = `rgb(0,${Math.round(cfg.glassRough * 255) | 0},0)`;
    rc.fillRect(px0 + 2, py0 + 2, Math.max(1, wpx - 4), Math.max(1, hpx - 4));

    // AO: recessed windows darker (red)
    ac.fillStyle = `rgb(${Math.round(150) | 0},255,255)`;
    ac.fillRect(px0 + 2, py0 + 2, Math.max(1, wpx - 4), Math.max(1, hpx - 4));
  }

  // floor spandrel shadow line (separates floors)
  alb.strokeStyle = 'rgba(0,0,0,0.18)';
  alb.lineWidth = 2;
  alb.beginPath(); alb.moveTo(0, H); alb.lineTo(W, H); alb.stroke();

  // ---- normal map -----------------------------------------------------------
  const normCanvas = sobelNormal(heightC.c);

  // ---- emissive lit-window variants ----------------------------------------
  const lits = [];
  for (let v = 0; v < LIT_VARIANTS; v++) {
    const lc = makeCanvas(W, H);
    const g = lc.ctx;
    g.fillStyle = '#04060a';
    g.fillRect(0, 0, W, H);
    // light a subset of window panes (per column per variant)
    for (let c = 0; c < cols; c++) {
      const wr = winRects[c];
      const px0 = Math.round(wr.u0 * W), px1 = Math.round(wr.u1 * W);
      const py0 = Math.round((1 - wr.v1) * H), py1 = Math.round((1 - wr.v0) * H);
      const wpx = Math.max(2, px1 - px0), hpx = Math.max(2, py1 - py0);
      // subdivide the column vertically into panes
      const panes = style === 'commercial' ? 3 : 2;
      for (let p = 0; p < panes; p++) {
        const lit = hash3(seed, c * 7 + v * 13, p * 5, style.charCodeAt(0)) > 0.26;
        if (!lit) continue;
        const pyy = py0 + Math.round((p / panes) * hpx);
        const phh = Math.max(2, Math.round(hpx / panes));
        // warm glow, slightly varied brightness (kept saturated for ACES)
        const bright = 0.8 + hash2(seed + v, c, p) * 0.2;
        const gx = cfg.litWarm ? (255 - 20 * hash2(seed + v, c, p)) : 240;
        const gr = cfg.litWarm ? (176 + 34 * hash2(seed + v, c, p)) : 205;
        const bl = cfg.litWarm ? (70 + 26 * hash2(seed + v, c, p)) : 190;
        const grad = g.createLinearGradient(0, pyy, 0, pyy + phh);
        grad.addColorStop(0, `rgba(${gx | 0},${gr | 0},${bl | 0},${bright})`);
        grad.addColorStop(1, `rgba(${(gx * 0.8) | 0},${(gr * 0.68) | 0},${(bl * 0.55) | 0},${bright})`);
        g.fillStyle = grad;
        g.fillRect(px0 + 1, pyy + 1, Math.max(1, wpx - 2), phh - 1);
      }
    }
    lits.push(lc);
  }

  return {
    albedo: toTex(albedo.c, true),
    normal: toTex(normCanvas, false),
    rough: toTex(roughC, false),
    ao: toTex(aoC, false),
    lit: lits.map(t => toTex(t, true)),
    metalness: cfg.metalness,
  };
}

// ---- rooftop (concrete) tile ------------------------------------------------
function buildRoofTile(seed) {
  const W = 256, H = 256;
  const { c, ctx } = makeCanvas(W, H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const n = hash2(seed + 99, x, y);
      let v = 132 + (n - 0.5) * 26;
      // seam lines every ~64px
      if ((x % 64) < 2 || (y % 64) < 2) v -= 16;
      ctx.fillStyle = `rgb(${v | 0},${(v + 4) | 0},${(v - 8) | 0})`;
      ctx.fillRect(x, y, 1, 1);
    }
  }
  return toTex(c, true);
}

// ---------------------------------------------------------------------------
// Geometry builders. Each building is built as its own small non-indexed
// BufferGeometry (walls + roof), merged later into style buckets.
// ---------------------------------------------------------------------------
function bakeGeo(g) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(g.pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(g.nrm, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(g.uv, 2));
  // uv1 copy for aoMap
  geo.setAttribute('uv1', new THREE.Float32BufferAttribute(g.uv.slice(), 2));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(g.col, 3));
  return geo;
}

function pushTri(g, a, b, c, nrm, col, ua, ub, uc) {
  const pa = [a[0], a[1], a[2]], pb = [b[0], b[1], b[2]], pc = [c[0], c[1], c[2]];
  for (const p of [pa, pb, pc]) { g.pos.push(p[0], p[1], p[2]); }
  for (let i = 0; i < 3; i++) g.nrm.push(nrm[0], nrm[1], nrm[2]);
  const uvs = [ua, ub, uc];
  for (const u of uvs) g.uv.push(u[0], u[1]);
  for (let i = 0; i < 3; i++) g.col.push(col, col, col);
}

// Generic face from corners [bl,br,tr,tl]; outward normal decides winding.
function addFace(g, corners, nrm, tint, uAxis, uBase, TILEW, vScale) {
  const flip = (nrm[0] + nrm[1] + nrm[2]) < 0;
  const uvOf = ci => {
    const c = corners[ci];
    const u = ((c[0] * uAxis[0] + c[2] * uAxis[1]) - uBase) / TILEW;
    // v uses y for walls (vScale=1/floorH) OR an explicit axis passed by caller.
    const v = Array.isArray(vScale)
      ? ((c[0] * vScale[0] + c[2] * vScale[1]) + vScale[2]) / 8.0
      : c[1] * vScale;
    return [u, v];
  };
  const tri = (a, b, cc) => pushTri(g, corners[a], corners[b], corners[cc], nrm, tint, uvOf(a), uvOf(b), uvOf(cc));
  if (flip) { tri(0, 2, 1); tri(0, 3, 2); }
  else { tri(0, 1, 2); tri(0, 2, 3); }
}

function buildWalls(w, d, H, floorH, tint) {
  const g = { pos: [], nrm: [], uv: [], col: [] };
  const wx = w / 2, dz = d / 2;
  // +z
  addFace(g,
    [[-wx, 0, dz], [wx, 0, dz], [wx, H, dz], [-wx, H, dz]],
    [0, 0, 1], tint, [1, 0], -wx, FACADE_TILE_W, 1 / floorH);
  // -z
  addFace(g,
    [[-wx, 0, -dz], [wx, 0, -dz], [wx, H, -dz], [-wx, H, -dz]],
    [0, 0, -1], tint, [1, 0], -wx, FACADE_TILE_W, 1 / floorH);
  // +x
  addFace(g,
    [[wx, 0, -dz], [wx, 0, dz], [wx, H, dz], [wx, H, -dz]],
    [1, 0, 0], tint, [0, 1], -dz, FACADE_TILE_W, 1 / floorH);
  // -x
  addFace(g,
    [[-wx, 0, -dz], [-wx, 0, dz], [-wx, H, dz], [-wx, H, -dz]],
    [-1, 0, 0], tint, [0, 1], -dz, FACADE_TILE_W, 1 / floorH);
  return bakeGeo(g);
}

function addBox(g, cx, cz, baseY, sx, sy, sz, tint) {
  // top face + 4 sides (bottom hidden). v axis = world metres mapped to concrete tile.
  const wx = sx / 2, hy = sy / 2, dz = sz / 2;
  // sides use vertical UV
  addFace(g,
    [[cx - wx, baseY, cz + dz], [cx + wx, baseY, cz + dz], [cx + wx, baseY + sy, cz + dz], [cx - wx, baseY + sy, cz + dz]],
    [0, 0, 1], tint, [1, 0], cx - wx, sx, 1 / sy);
  addFace(g,
    [[cx - wx, baseY, cz - dz], [cx + wx, baseY, cz - dz], [cx + wx, baseY + sy, cz - dz], [cx - wx, baseY + sy, cz - dz]],
    [0, 0, -1], tint, [1, 0], cx - wx, sx, 1 / sy);
  addFace(g,
    [[cx + wx, baseY, cz - dz], [cx + wx, baseY, cz + dz], [cx + wx, baseY + sy, cz + dz], [cx + wx, baseY + sy, cz - dz]],
    [1, 0, 0], tint, [0, 1], cz - dz, sz, 1 / sy);
  addFace(g,
    [[cx - wx, baseY, cz - dz], [cx - wx, baseY, cz + dz], [cx - wx, baseY + sy, cz + dz], [cx - wx, baseY + sy, cz - dz]],
    [-1, 0, 0], tint, [0, 1], cz - dz, sz, 1 / sy);
  // top
  addFace(g,
    [[cx - wx, baseY + sy, cz - dz], [cx + wx, baseY + sy, cz - dz], [cx + wx, baseY + sy, cz + dz], [cx - wx, baseY + sy, cz + dz]],
    [0, 1, 0], tint * 0.9, [1, 0], cx - wx, sx, [0, 1, -cz + dz]);
}

function buildRoof(w, d, H, rng) {
  const g = { pos: [], nrm: [], uv: [], col: [] };
  const wx = w / 2, dz = d / 2;
  // slab
  addFace(g,
    [[-wx, H, -dz], [wx, H, -dz], [wx, H, dz], [-wx, H, dz]],
    [0, 1, 0], 1.0, [1, 0], -wx, 8.0, [0, 1, -dz]);
  // parapet ring
  const p = 1.3;
  addFace(g, [[-wx, H, dz], [wx, H, dz], [wx, H + p, dz], [-wx, H + p, dz]], [0, 0, 1], 1.05, [1, 0], -wx, FACADE_TILE_W, 1 / 3.2);
  addFace(g, [[-wx, H, -dz], [wx, H, -dz], [wx, H + p, -dz], [-wx, H + p, -dz]], [0, 0, -1], 1.05, [1, 0], -wx, FACADE_TILE_W, 1 / 3.2);
  addFace(g, [[wx, H, -dz], [wx, H, dz], [wx, H + p, dz], [wx, H + p, -dz]], [1, 0, 0], 1.05, [0, 1], -dz, FACADE_TILE_W, 1 / 3.2);
  addFace(g, [[-wx, H, -dz], [-wx, H, dz], [-wx, H + p, dz], [-wx, H + p, -dz]], [-1, 0, 0], 1.05, [0, 1], -dz, FACADE_TILE_W, 1 / 3.2);

  // rooftop AC units
  const nAC = 2 + Math.floor((w * d) / 420);
  for (let i = 0; i < nAC; i++) {
    const ax = -wx + 4 + rng() * (w - 8);
    const az = -dz + 4 + rng() * (d - 8);
    addBox(g, ax, az, H, 1.8 + rng() * 0.8, 1.2 + rng() * 0.6, 1.6 + rng() * 0.9, 0.5);
  }
  return bakeGeo(g);
}

// ---------------------------------------------------------------------------
// City layout. Reads world.zones defensively; falls back to a deterministic
// default downtown master plan when zones are absent/empty.
// ---------------------------------------------------------------------------
function collectLots(world) {
  const rng = typeof world.rng === 'function' ? world.rng : mulberry32(Number.isFinite(world.meta && world.meta.seed) ? world.meta.seed : 1337);
  const lots = [];

  const zones = (Array.isArray(world.zones) && world.zones.length) ? world.zones : null;
  if (zones) {
    for (const z of zones) {
      if (!z || typeof z.x !== 'number' || typeof z.z !== 'number') continue;
      const type = String(z.type || '').toLowerCase();
      let style = 'residential';
      if (type.includes('com')) style = 'commercial';
      else if (type.includes('ind')) style = 'industrial';
      const density = Number.isFinite(z.density) ? z.density : 3;
      lots.push({ cx: z.x, cz: z.z, style, density, fromZone: true });
    }
    return lots;
  }

  // ---- default downtown master plan (zoning not landed yet) ---------------
  const OFFSETS = [60, -60, 180, -180, 300, -300, 420, -420, 540, -540];
  const HBLK = 44;              // buildable half-width inside a block (m)
  for (const bx of OFFSETS) {
    for (const bz of OFFSETS) {
      const d = Math.hypot(bx, bz);
      let district, grid;
      if (d < 170) { district = 'core'; grid = 3; }
      else if (d < 330) { district = 'mid'; grid = 3; }
      else { district = 'edge'; grid = 2; }

      const cell = (HBLK * 2) / grid;
      // deterministic per-block variation
      const blockSeed = Math.round(bx * 13 + bz * 7 + 1);
      for (let gy = 0; gy < grid; gy++) {
        for (let gx = 0; gx < grid; gx++) {
          const fillP = district === 'core' ? 0.84 : district === 'mid' ? 0.70 : 0.52;
          if (rng() > fillP) continue;

          // lot center inside this cell with jitter
          const cx = bx - HBLK + gx * cell + cell / 2 + (rng() - 0.5) * cell * 0.3;
          const cz = bz - HBLK + gy * cell + cell / 2 + (rng() - 0.5) * cell * 0.3;

          // style & density by district
          let style, density;
          if (district === 'core') { style = 'commercial'; density = 4 + Math.round(rng() * 2); }
          else if (district === 'mid') {
            const com = rng() < 0.38;
            style = com ? 'commercial' : 'residential';
            density = com ? 3 + Math.round(rng() * 2) : 2 + Math.round(rng() * 3);
          } else {
            const ind = rng() < 0.34;
            style = ind ? 'industrial' : 'residential';
            density = 1 + Math.round(rng() * (ind ? 1 : 3));
          }
          // dense commercial spine along the eastern avenue so the east-facing
          // street/night camera reads a glowing high-rise skyline.
          if ((district === 'mid' || district === 'edge') && cx > 80 && Math.abs(cz) < 170 && rng() < 0.5) {
            style = 'commercial';
            density = 4 + Math.round(rng());
          }
          lots.push({ cx, cz, style, density, blockSeed, district });
        }
      }
    }
  }
  return lots;
}

function planBuilding(lot, rng) {
  const { style, density } = lot;
  let w, d, h;

  if (style === 'commercial') {
    // towers / slabs
    w = 12 + rng() * (6 + density * 3);
    d = w * (0.7 + rng() * 0.6);
    const anchor = lot.district === 'core' && Math.hypot(lot.cx, lot.cz) < 150 && rng() < 0.30;
    h = anchor ? 120 + rng() * 80 : (34 + density * 12) + rng() * (20 + density * 6);
  } else if (style === 'residential') {
    w = 14 + rng() * (16 + density * 4);
    d = w * (0.8 + rng() * 0.7);
    h = (9 + density * 7) + rng() * (16 + density * 5);
  } else { // industrial warehouse
    w = 24 + rng() * 20;
    d = 16 + rng() * 14;
    h = 8 + rng() * 8;
  }
  return { w: Math.round(w), d: Math.round(d), h: Math.round(h) };
}

// ---------------------------------------------------------------------------
// Module API
// ---------------------------------------------------------------------------
export function init(world) {
  try {
    if (!world || !world.scene || typeof world.heightAt !== 'function') return;
    const seed = Number.isFinite(world.meta && world.meta.seed) ? world.meta.seed : 1337;
    const rng = typeof world.rng === 'function' ? world.rng : mulberry32(seed);

    // ---- style tiles + materials ------------------------------------------
    const wallBuckets = {};
    const roofs = [];
    const styleMats = {};

    for (const style of ['commercial', 'residential', 'industrial']) {
      const tiles = buildStyleTiles((seed ^ 0x5f3759df) + style.charCodeAt(0), style);
      styleMats[style] = [];
      for (let v = 0; v < LIT_VARIANTS; v++) {
        const mat = new THREE.MeshStandardMaterial({
          map: tiles.albedo,
          normalMap: tiles.normal,
          roughnessMap: tiles.rough,
          aoMap: tiles.ao,
          metalness: tiles.metalness,
          vertexColors: true,
          emissive: new THREE.Color(1.0, 0.82, 0.55),
          emissiveMap: tiles.lit[v],
          emissiveIntensity: 0,
        });
        mat.userData.style = style;
        styleMats[style].push(mat);
        state.materials.push(mat);
      }
    }

    const roofTex = buildRoofTile(seed);
    const roofMat = new THREE.MeshStandardMaterial({
      map: roofTex, roughness: 0.9, metalness: 0.1, vertexColors: true,
    });

    // ---- place buildings ---------------------------------------------------
    const lots = collectLots(world);
    let counter = 0;

    for (const lot of lots) {
      if (!Number.isFinite(lot.cx) || !Number.isFinite(lot.cz)) continue;
      const { w, d, h } = planBuilding(lot, rng);
      const style = lot.style || 'residential';
      const floorH = FLOOR_H[style] || 3.1;

      const groundY = world.heightAt(lot.cx, lot.cz) - 0.4;
      // subtle per-building tint so blocks are never uniform clones
      const tint = 0.92 + rng() * 0.16;

      // assign a lit-window variant
      const variant = Math.floor(rng() * LIT_VARIANTS);

      // build wall geometry
      const wallGeo = buildWalls(w, d, h, floorH, tint);
      wallGeo.computeBoundingSphere();

      // translate from local box (base at y=0) to world position
      // Bucket by style × lit-variant × spatial region so merged meshes have
      // tight bounding volumes and get frustum-culled when off-screen/behind cam.
      const REGION = 400;
      const rx = Math.max(0, Math.min(2, Math.floor((lot.cx + 600) / REGION)));
      const rz = Math.max(0, Math.min(2, Math.floor((lot.cz + 600) / REGION)));
      const key = `${style}_${variant}_${rx}_${rz}`;
      if (!wallBuckets[key]) wallBuckets[key] = [];
      wallGeo.translate(lot.cx, groundY, lot.cz);
      wallBuckets[key].push(wallGeo);

      // roof geometry
      const roofGeo = buildRoof(w, d, h, rng);
      roofGeo.computeBoundingSphere();
      roofGeo.translate(lot.cx, groundY, lot.cz);
      roofs.push(roofGeo);

      // record + emit
      const rec = {
        id: `b${counter++}`, x: lot.cx, z: lot.cz, y: groundY,
        type: style, w, d, h,
        density: Number.isFinite(lot.density) ? lot.density : 3,
        floors: Math.max(1, Math.round(h / floorH)),
        units: Math.max(1, Math.round((w * d) / 90)),
      };
      world.buildings.push(rec);
      bus.emit('building:add', { b: rec });
    }

    // ---- merge + add meshes ------------------------------------------------
    const groups = [];
    for (const key of Object.keys(wallBuckets)) {
      const parts = key.split('_');
      const style = parts[0];
      const vIdx = Number(parts[1]);
      const geos = wallBuckets[key];
      if (!geos.length || !styleMats[style]) continue;
      let merged = null;
      try {
        merged = mergeGeometries(geos);
      } catch (e) { merged = null; }
      if (merged) {
        const mesh = new THREE.Mesh(merged, styleMats[style][vIdx]);
        mesh.castShadow = true;
        mesh.frustumCulled = true;
        mesh.matrixAutoUpdate = false;
        world.scene.add(mesh);
        groups.push(mesh);
      } else {
        // fallback: individual meshes (rare)
        for (const g of geos) {
          const m = new THREE.Mesh(g, styleMats[style][vIdx]);
          m.castShadow = true;
          world.scene.add(m);
          groups.push(m);
        }
      }
    }

    // roof mesh
    if (roofs.length) {
      let mergedRoof = null;
      try { mergedRoof = mergeGeometries(roofs); } catch (e) { mergedRoof = null; }
      if (mergedRoof) {
        const m = new THREE.Mesh(mergedRoof, roofMat);
        m.castShadow = true;
        m.matrixAutoUpdate = false;
        world.scene.add(m);
        groups.push(m);
      }
    }

    state.groups = groups;
    state.built = true;
  } catch (e) {
    console.error('[buildings] init error', e);
  }
}

// Day/night: fade lit windows in as dusk falls / dawn rises.
export function update(dt, world) {
  try {
    if (!state.built || !world.meta) return;
    const sec = ((world.meta.timeOfDaySec % 86400) + 86400) % 86400;
    const DUSK_A = 17.5 * 3600, DUSK_B = 20.0 * 3600;
    const DAWN_A = 6.3 * 3600, DAWN_B = 4.7 * 3600; // dusk->dawn ramp reversed
    let f = 0;
    if (sec > DUSK_A) {
      f = clamp01((sec - DUSK_A) / (DUSK_B - DUSK_A));
    }
    const dawnFade = clamp01((DAWN_A - sec) / (DAWN_A - DAWN_B)); // 1 while night
    f = Math.max(f, dawnFade);
    if (f !== state.nightFactor) {
      state.nightFactor = f;
      for (const m of state.materials) m.emissiveIntensity = NIGHT_LIT * f;
    }
  } catch (e) { /* isolation */ }
}

export function showcase(container) {
  try {
    if (!container || typeof container.appendChild !== 'function') return;
    const div = document.createElement('div');
    div.style.cssText =
      'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;' +
      'font-family:system-ui,sans-serif;color:#e8ecef;background:linear-gradient(#1b2a3a,#0c1420);';
    div.innerHTML =
      '<div style="text-align:center">' +
      '<h2 style="margin:0 0 6px;font-weight:600">Buildings</h2>' +
      '<p style="margin:0;opacity:.85">procedural PBR facades · merged geometry · day/night lit windows</p>' +
      `<p style="margin:8px 0 0;font-size:12px;opacity:.6">${state.built ? state.groups.length + ' draw groups, warm night glow' : 'uninitialized'}</p>` +
      '</div>';
    container.appendChild(div);
  } catch (_) {}
}
