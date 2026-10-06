// CC0 asset + PBR material factory. Procedural by default; photographic
// Poly Haven / ambientCG textures can be dropped into /assets/<name>.* and are
// picked up automatically (same interface, so they're swappable).
// PBR invariants: albedo = sRGB, data maps (rough/metal/normal) = linear.
import * as THREE from 'three';

function smooth(t) { return t * t * (3 - 2 * t); }

// 2D value noise on an integer lattice, seed-mixed.
function valueNoise(x, y, seed) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const h = (a, b) => {
    let n = (Math.imul(a, 374761393) + Math.imul(b, 668265263) + Math.imul(seed, 974634799)) >>> 0;
    n = (n ^ (n >>> 13)) >>> 0; n = Math.imul(n, 0x5bd1e995) >>> 0; n ^= n >>> 15;
    return (n >>> 0) / 4294967296;
  };
  const a = h(ix, iy), b = h(ix + 1, iy), c = h(ix, iy + 1), d = h(ix + 1, iy + 1);
  const u = smooth(fx), v = smooth(fy);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
function fbm(x, y, seed, oct = 4) {
  let sum = 0, amp = 0.5, f = 1, norm = 0;
  for (let i = 0; i < oct; i++) { sum += amp * valueNoise(x * f, y * f, seed + i * 101); norm += amp; amp *= 0.5; f *= 2; }
  return sum / norm;
}

const S = 256;
function canvas() { const c = document.createElement('canvas'); c.width = c.height = S; return c; }

// Returns {map, roughnessMap?, normalMap?} for a surface name, procedural.
function buildProcedural(name, seed) {
  const c = canvas(); const g = c.getContext('2d');
  const img = g.createImageData(S, S); const px = img.data;
  const rc = canvas(); const rg = rc.getContext('2d'); const rimg = rg.createImageData(S, S); const rp = rimg.data;

  const set = (i, r, gg, b) => { px[i] = r; px[i + 1] = gg; px[i + 2] = b; px[i + 3] = 255; };

  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = (y * S + x) * 4;
      const u = x / S, v = y / S;
      const n = fbm(u * 8, v * 8, seed, 5);
      const fine = valueNoise(u * 64, v * 64, seed + 7);
      let r = 128, gg = 128, b = 128, rough = 200;
      switch (name) {
        case 'asphalt': {
          const base = 40 + n * 22 + fine * 14;
          r = base; gg = base; b = base + 2; rough = 210 - fine * 40; break;
        }
        case 'concrete': {
          const base = 150 + n * 40 + fine * 12;
          r = base; gg = base - 2; b = base - 6; rough = 225 - fine * 30; break;
        }
        case 'grass': {
          const m = fbm(u * 12, v * 12, seed + 3, 5);
          r = 60 + m * 40; gg = 96 + m * 70 + fine * 20; b = 44 + m * 30; rough = 235; break;
        }
        case 'dirt': {
          const m = fbm(u * 10, v * 10, seed + 5, 5);
          r = 120 + m * 50; gg = 92 + m * 40; b = 62 + m * 28; rough = 240; break;
        }
        case 'roof': {
          const m = fbm(u * 6, v * 6, seed + 9, 4);
          r = 90 + m * 50; gg = 84 + m * 46; b = 80 + m * 44; rough = 220 - fine * 20; break;
        }
        case 'water': {
          const m = fbm(u * 5 + 0.3, v * 5, seed + 11, 4);
          r = 30 + m * 20; gg = 60 + m * 30; b = 84 + m * 40; rough = 30; break;
        }
        case 'facade': {
          // window grid: bright glass at night handled via emissive on the building side
          const winX = Math.floor(u * 16), winY = Math.floor(v * 24);
          const inWin = (u * 16 % 1) > 0.22 && (u * 16 % 1) < 0.78 && (v * 24 % 1) > 0.28 && (v * 24 % 1) < 0.72;
          const base = 120 + n * 30;
          if (inWin) { r = 70; gg = 92; b = 110; } else { r = base; gg = base - 4; b = base - 10; }
          rough = inWin ? 40 : 200; break;
        }
        default: { r = 128; gg = 128; b = 128; rough = 200; }
      }
      set(i, r, gg, b);
      rp[i] = rp[i + 1] = rp[i + 2] = rough; rp[i + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  rg.putImageData(rimg, 0, 0);
  return { map: c, roughnessMap: rc };
}

export class Assets {
  constructor(three = THREE) {
    this.three = three;
    this.base = '/assets/';
    this._tex = new Map();
    this._mat = new Map();
  }
  _toTexture(canvasEl, srgb) {
    const t = new this.three.CanvasTexture(canvasEl);
    t.wrapS = t.wrapT = this.three.RepeatWrapping;
    t.colorSpace = srgb ? this.three.SRGBColorSpace : this.three.NoColorSpace;
    t.anisotropy = 8;
    t.needsUpdate = true;
    return t;
  }
  // Cached surface texture set. name e.g. 'asphalt'.
  surface(name, seed = 1) {
    const key = `${name}:${seed}`;
    if (this._tex.has(key)) return this._tex.get(key);
    const { map, roughnessMap } = buildProcedural(name, seed);
    const set = {
      map: this._toTexture(map, true),
      roughnessMap: this._toTexture(roughnessMap, false),
    };
    this._tex.set(key, set);
    return set;
  }
  // PBR material for a surface, with overrides.
  material(name, params = {}, seed = 1) {
    const key = `${name}:${seed}:${JSON.stringify(params)}`;
    if (this._mat.has(key)) return this._mat.get(key);
    const s = this.surface(name, seed);
    const m = new this.three.MeshStandardMaterial({
      map: s.map,
      roughnessMap: s.roughnessMap,
      roughness: 1,
      metalness: 0,
      ...params,
    });
    this._mat.set(key, m);
    return m;
  }
  // Load a photographic texture from /assets/<name>.<ext> if present; else fallback procedural.
  async loadPhoto(name, { srgb = true, scale = 1 } = {}) {
    for (const ext of ['jpg', 'jpeg', 'png', 'webp']) {
      const url = this.base + `${name}.${ext}`;
      try {
        const res = await fetch(url);
        if (!res.ok) continue;
        const blob = await res.blob();
        const bmp = await createImageBitmap(blob);
        const t = new this.three.Texture(bmp);
        t.wrapS = t.wrapT = this.three.RepeatWrapping;
        t.colorSpace = srgb ? this.three.SRGBColorSpace : this.three.NoColorSpace;
        t.anisotropy = 8;
        t.repeat.set(scale, scale);
        t.needsUpdate = true;
        this._tex.set(`photo:${name}`, t);
        return t;
      } catch { /* try next ext */ }
    }
    return null;
  }
}
